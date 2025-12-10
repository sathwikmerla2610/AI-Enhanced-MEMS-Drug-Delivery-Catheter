import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)

# ============================================================================
# SECTION 1: PHYSICAL DEVICE PARAMETERS
# ============================================================================

@dataclass
class DeviceParameters:
    """Complete physical specifications of the MEMS Device"""
    parylene_membrane_thickness: float = 5e-6
    pt_thickness: float = 200e-9
    bellows_diameter: float = 6e-3
    bellows_height_relaxed: float = 4e-3
    bellows_height_expanded: float = 8e-3
    electrode_length: float = 3e-3
    faraday_constant: float = 96485.0
    gas_constant: float = 8.314
    temperature: float = 310.15
    nominal_current: float = 1.0e-3
    max_current: float = 2.0e-3
    max_pressure: float = 200000

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class Config:
    PHYSICS = DeviceParameters()
    MAX_CURRENT = PHYSICS.max_current

    NOISE_ENABLED = True

    # Optimized for performance while maintaining accuracy
    TRAINING_SAMPLES = 50000

    DT = 0.05
    DURATION = 60

    MU_BLOOD = 0.0035
    GAMMA = 0.05
    NOMINAL_ID = 500e-6
    LENGTH = 0.1
    SPIRAL_FACTOR = 1.3

    PRELOAD_PRESSURE = 2500
    P_ATM = 101325
    INIT_GAS_VOL = 20e-6
    RESERVOIR_VOL = 50e-6

    Z_VAL = 2.67
    R0_RTD = 1000.0
    ALPHA_RTD = 0.00385
    HF_A = 1.5
    HF_B = 4.0
    HF_N = 0.45
    GLUCOSE_SENS = 15.0

    R_HYDRO = (128 * MU_BLOOD * LENGTH * SPIRAL_FACTOR) / (np.pi * NOMINAL_ID**4 + 1e-12)
    P_CAPILLARY = (4 * GAMMA) / (NOMINAL_ID + 1e-12)
    HF_AREA = np.pi * (NOMINAL_ID/2)**2 + 1e-12
    TO_UL_MIN = 60 * 1e9

# ============================================================================
# SECTION 3: PHYSICS ENGINE
# ============================================================================

class BellowsPump:
    def __init__(self):
        self.params = Config.PHYSICS
        self.total_chamber_vol = Config.INIT_GAS_VOL + Config.RESERVOIR_VOL
        p_initial_abs = Config.P_ATM + Config.PRELOAD_PRESSURE
        self.n_gas_moles = (p_initial_abs * Config.INIT_GAS_VOL) / (
            self.params.gas_constant * self.params.temperature
        )
        self.vol_liquid = Config.RESERVOIR_VOL
        self.vol_gas = Config.INIT_GAS_VOL
        self.pressure_internal = p_initial_abs
        self.flow_rate = 0.0
        self.bellows_height = self.params.bellows_height_relaxed

    def update(self, current_I, dt, p_external):
        moles_new = (current_I * dt) / (Config.Z_VAL * self.params.faraday_constant)
        self.n_gas_moles += moles_new

        self.vol_gas = max(self.total_chamber_vol - self.vol_liquid, 1e-12)
        radius = self.params.bellows_diameter / 2
        self.bellows_height = self.vol_gas / (np.pi * radius**2 + 1e-12)

        self.pressure_internal = (
            self.n_gas_moles * self.params.gas_constant * self.params.temperature
        ) / self.vol_gas
        self.pressure_internal = min(self.pressure_internal, Config.P_ATM + self.params.max_pressure)

        P_gauge = self.pressure_internal - Config.P_ATM
        P_net = P_gauge - p_external - Config.P_CAPILLARY

        if self.vol_liquid <= 0 or P_net <= 0:
            self.flow_rate = 0.0
        else:
            self.flow_rate = P_net / Config.R_HYDRO

        self.vol_liquid = max(0.0, self.vol_liquid - (self.flow_rate * dt))
        return P_gauge

# ============================================================================
# SECTION 4: SENSORS
# ============================================================================

class SensorSuite:
    def __init__(self):
        self.prev_temp = 37.0

    def read(self, Q_true, T_fluid, C_glucose, dt):
        def noise(scale):
            return np.random.normal(0, scale) if Config.NOISE_ENABLED else 0.0

        R_meas = Config.R0_RTD * (1 + Config.ALPHA_RTD * (T_fluid - 25)) + noise(0.1)
        U = Q_true / Config.HF_AREA
        V_king = np.sqrt(Config.HF_A + Config.HF_B * (U ** Config.HF_N))
        temp_correction = 1.0 - 0.02 * (T_fluid - 37.0)
        dT_dt = np.clip((T_fluid - self.prev_temp) / dt, -2.0, 2.0)
        transient_error = -0.05 * dT_dt
        V_meas = (V_king * temp_correction) + transient_error + noise(0.002)
        I_meas = Config.GLUCOSE_SENS * C_glucose + noise(0.1)
        self.prev_temp = T_fluid
        return V_meas, R_meas, I_meas, dT_dt

class SmartCatheter:
    def __init__(self):
        self.pump = BellowsPump()
        self.sensors = SensorSuite()

    def get_sensor_readings(self, temp, glucose, dt):
        return self.sensors.read(self.pump.flow_rate, temp, glucose, dt)

    def step_physics(self, I_cmd, P_tissue, dt):
        P_gauge = self.pump.update(I_cmd, dt, P_tissue)
        return {
            'P_gauge': P_gauge,
            'Q_true': self.pump.flow_rate,
            'Vol_Res': self.pump.vol_liquid,
            'Bellows_H': self.pump.bellows_height
        }

def naive_kings_law(V_meas):
    val = np.maximum(0, (V_meas**2 - Config.HF_A) / Config.HF_B)
    return (val ** (1/Config.HF_N)) * Config.HF_AREA

# ============================================================================
# SECTION 5: AI MODELS (RandomForest vs Naive Only)
# ============================================================================

def generate_training_data_vectorized(n_samples, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(42)

    logger.info(f"Generating {n_samples:,} training samples...")

    P_net = random_state.uniform(0, 12000, n_samples)
    mu = random_state.uniform(0.003, 0.0045, n_samples)
    d_ID = np.maximum(random_state.normal(Config.NOMINAL_ID, 10e-6, n_samples), 100e-6)

    R_var = (128 * mu * Config.LENGTH * Config.SPIRAL_FACTOR) / (np.pi * d_ID**4 + 1e-12)
    Q_true = P_net / R_var

    T = random_state.uniform(36, 40, n_samples)
    dT_dt = random_state.uniform(-0.1, 0.1, n_samples)

    U = Q_true / Config.HF_AREA
    V_king = np.sqrt(Config.HF_A + Config.HF_B * (U ** Config.HF_N))
    temp_corr = 1.0 - 0.02 * (T - 37.0)
    V_meas = (V_king * temp_corr) - (0.05 * dT_dt)
    R_meas = Config.R0_RTD * (1 + Config.ALPHA_RTD * (T - 25))

    return pd.DataFrame({
        'Q_True': Q_true, 'V_HF': V_meas, 'R_RTD': R_meas, 'dT_dt': dT_dt
    })

def train_random_forest(X_train, y_train, X_test, y_test):
    logger.info("  Training Random Forest...")
    # Slightly optimized parameters for speed without losing much accuracy
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=20,
        min_samples_split=5, min_samples_leaf=2, max_features='sqrt', n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {'type': 'RandomForest', 'model': model}, np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred), y_pred

def predict_with_model(model_wrapper, X):
    model = model_wrapper.get('model')
    return model.predict(X)

def get_models():
    """Train RandomForest and compare with Naive baseline"""
    logger.info("=" * 80)
    logger.info("TRAINING AI MODEL vs NAIVE BASELINE")
    logger.info("=" * 80)

    n_samples = Config.TRAINING_SAMPLES
    df = generate_training_data_vectorized(n_samples, np.random.RandomState(42))

    X = df[['V_HF', 'R_RTD', 'dT_dt']]
    y = df['Q_True']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training: {len(X_train):,} | Test: {len(X_test):,}")

    models = {}

    # Random Forest
    rf_model, rf_rmse, rf_r2, rf_pred = train_random_forest(X_train, y_train, X_test, y_test)
    models['RandomForest'] = {'model': rf_model, 'rmse': rf_rmse, 'r2': rf_r2, 'pred': rf_pred}
    logger.info(f"  ✓ RandomForest: RMSE={rf_rmse*Config.TO_UL_MIN:.3f} μL/min, R²={rf_r2:.4f}")

    # Naive baseline
    y_naive = X_test['V_HF'].clip(lower=1e-6).apply(naive_kings_law)
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_naive))
    naive_r2 = r2_score(y_test, y_naive)
    models['Naive'] = {'rmse': naive_rmse, 'r2': naive_r2, 'pred': y_naive.values}
    logger.info(f"  ✓ Naive Model:  RMSE={naive_rmse*Config.TO_UL_MIN:.3f} μL/min, R²={naive_r2:.4f}")

    return models, X_test, y_test

# ============================================================================
# SECTION 6: CONTROL
# ============================================================================

class AdaptivePID:
    def __init__(self, base_Kp, base_Ki, base_Kd, limit):
        self.base_Kp, self.base_Ki, self.base_Kd = base_Kp, base_Ki, base_Kd
        self.Kp, self.Ki, self.Kd = base_Kp, base_Ki, base_Kd
        self.limit = limit
        self.prev_error = 0
        self.integral = 0

    def adapt_gains(self, target_flow_ul_min):
        if target_flow_ul_min > 100.0:
            self.Kp, self.Ki, self.Kd = self.base_Kp * 1.5, self.base_Ki * 1.2, self.base_Kd * 1.2
        else:
            self.Kp, self.Ki, self.Kd = self.base_Kp, self.base_Ki, self.base_Kd

    def compute(self, setpoint, measured, dt):
        error = setpoint - measured
        self.integral += error * dt
        if self.Ki > 0:
            self.integral = np.clip(self.integral, -self.limit / (self.Ki + 1e-12), self.limit / (self.Ki + 1e-12))
        derivative = (error - self.prev_error) / (dt + 1e-12)
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return np.clip(output, 0, self.limit)

# ============================================================================
# SECTION 7: COMPREHENSIVE VISUALIZATION
# ============================================================================

def plot_model_comparison(models, X_test, y_test):
    """Plot 1: Combined comparison - Random Forest vs Naive in ONE plot"""
    to_ul = Config.TO_UL_MIN

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('Direct Model Comparison: AI vs Naive Baseline', fontsize=18, fontweight='bold')

    y_test_ul = y_test.reset_index(drop=True) * to_ul
    max_val = y_test_ul.max() * 1.1

    # Specific colors
    colors = {'RandomForest': '#22c55e', 'Naive': '#ef4444'} # Green vs Red

    # Plot Naive first (background), then RF (foreground)
    for name in ['Naive', 'RandomForest']:
        data = models[name]
        pred_ul = pd.Series(data['pred']).reset_index(drop=True) * to_ul

        # Style adjustments for visibility
        alpha = 0.6 if name == 'RandomForest' else 0.4
        marker = 'o' if name == 'RandomForest' else 's'
        size = 30 if name == 'RandomForest' else 20

        ax.scatter(y_test_ul, pred_ul,
                   alpha=alpha,
                   c=colors[name],
                   s=size,
                   marker=marker,
                   edgecolors='none',
                   label=f'{name} (R²={data["r2"]:.4f})')

    # Ideal Line
    ax.plot([0, max_val], [0, max_val], 'k--', lw=3, label='Ideal Prediction (1:1)')

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('True Flow Rate (μL/min)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Flow Rate (μL/min)', fontweight='bold', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotation box
    rf_rmse = models['RandomForest']['rmse']
    naive_rmse = models['Naive']['rmse']
    improvement = (naive_rmse - rf_rmse) / naive_rmse * 100

    textstr = f'Random Forest Improvement:\n{improvement:.1f}% Accuracy Gain'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold', color='green')

    plt.tight_layout()
    plt.show()

def plot_model_metrics_comparison(models):
    """Plot 2: Bar chart comparing model metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')

    names = ['RandomForest', 'Naive']
    colors = ['#22c55e', '#ef4444'] # Green, Red

    # R² Score
    r2_scores = [models[n]['r2'] for n in names]
    bars1 = axes[0].bar(names, r2_scores, color=colors, edgecolor='black')
    axes[0].set_ylabel('R² Score', fontweight='bold')
    axes[0].set_title('Prediction Accuracy (R²)', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', fontweight='bold')

    # RMSE
    rmse_scores = [models[n]['rmse'] * Config.TO_UL_MIN for n in names]
    bars2 = axes[1].bar(names, rmse_scores, color=colors, edgecolor='black')
    axes[1].set_ylabel('RMSE (μL/min)', fontweight='bold')
    axes[1].set_title('Error (RMSE) - Lower is Better', fontweight='bold')
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', ha='center', fontweight='bold')

    # Improvement over Naive
    naive_rmse = models['Naive']['rmse']
    rf_imp = (naive_rmse - models['RandomForest']['rmse']) / naive_rmse * 100
    improvements = [rf_imp, 0]

    bars3 = axes[2].bar(names, improvements, color=colors, edgecolor='black')
    axes[2].set_ylabel('Improvement (%)', fontweight='bold')
    axes[2].set_title('Improvement Over Naive', fontweight='bold')
    axes[2].axhline(y=0, color='black', linestyle='-', lw=1)

    axes[2].text(0, rf_imp + 1, f'+{rf_imp:.1f}%', ha='center', fontweight='bold', color='green', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_error_distribution(models, y_test):
    """Plot 3: Error distribution for each model"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')

    y_test_np = y_test.reset_index(drop=True).values * Config.TO_UL_MIN
    colors = {'RandomForest': '#22c55e', 'Naive': '#ef4444'}

    for idx, name in enumerate(['RandomForest', 'Naive']):
        ax = axes[idx]
        data = models[name]
        errors = (data['pred'] * Config.TO_UL_MIN) - y_test_np

        ax.hist(errors, bins=50, color=colors[name], edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', lw=2, label='Zero Error')
        ax.set_xlabel('Prediction Error (μL/min)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{name}\nStd Dev: {np.std(errors):.3f} μL/min', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_simulation_loop(model, pressure_case_mmHg=None):
    """Run physics simulation with AI feedback"""
    time_steps = np.arange(0, Config.DURATION + Config.DT, Config.DT)

    catheter = SmartCatheter()
    pid = AdaptivePID(base_Kp=1e-5, base_Ki=5e-6, base_Kd=1e-7, limit=Config.MAX_CURRENT)
    log_data = []

    for t in time_steps:
        glucose = 5.0 + 6.0 * np.exp(-((t - 30)**2) / (2 * 5**2))
        if Config.NOISE_ENABLED:
            glucose += np.random.normal(0, 0.1)

        temp = 37.0 + 0.5 * np.sin(0.2 * t)
        if Config.NOISE_ENABLED:
            temp += np.random.normal(0, 0.05)

        P_tissue = pressure_case_mmHg * 133.322 if pressure_case_mmHg else 1000 + 200 * np.sin(1.0 * t)
        target_Q_SI = 5e-9 if glucose > 8.0 else 1e-9

        V, R, I_gluc, dT_dt = catheter.get_sensor_readings(temp, glucose, Config.DT)

        input_feat = pd.DataFrame([[V, R, dT_dt]], columns=['V_HF', 'R_RTD', 'dT_dt'])
        Q_pred_SI = predict_with_model(model, input_feat)[0]

        meas_ul_min = max(0, Q_pred_SI * Config.TO_UL_MIN) if Q_pred_SI * Config.TO_UL_MIN >= 0.5 else 0.0
        targ_ul_min = target_Q_SI * Config.TO_UL_MIN

        pid.adapt_gains(targ_ul_min)
        I_cmd = pid.compute(targ_ul_min, meas_ul_min, Config.DT)
        phys_state = catheter.step_physics(I_cmd, P_tissue, Config.DT)

        log_data.append({
            'time': t, 'Q_true': phys_state['Q_true'], 'Q_pred': Q_pred_SI,
            'Q_target': target_Q_SI, 'P_bellows': phys_state['P_gauge'],
            'I_pump': I_cmd, 'Vol_Res': phys_state['Vol_Res'],
            'Glucose': glucose, 'Temp': temp, 'P_tissue': P_tissue,
            'Q_Naive': naive_kings_law(V), 'Bellows_H': phys_state['Bellows_H'],
            'V_HF': V, 'R_RTD': R
        })

    return pd.DataFrame(log_data)


def plot_comprehensive_dashboard(results_dict):
    """Plot 4: Main simulation dashboard"""
    to_ul = Config.TO_UL_MIN
    main_df = results_dict['Dynamic']

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Smart Catheter Comprehensive Dashboard - Random Forest Control', fontsize=20, fontweight='bold')

    # Row 1: Flow Control
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(main_df['time'], main_df['Q_target'] * to_ul, 'k--', lw=2.5, label='Target')
    ax1.plot(main_df['time'], main_df['Q_true'] * to_ul, 'b-', lw=2, label='True Flow')
    ax1.plot(main_df['time'], main_df['Q_pred'] * to_ul, color='#22c55e', linestyle=':', lw=2.5, label='RF Predicted')
    ax1.fill_between(main_df['time'], main_df['Q_true'] * to_ul, alpha=0.1, color='blue')
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_ylabel('Flow Rate (μL/min)', fontweight='bold')
    ax1.set_title('Closed-Loop Flow Control', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)

    # Backpressure comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    pressures = [k for k in results_dict.keys() if k != 'Dynamic']
    flows = [results_dict[p]['Q_true'].mean() * to_ul for p in pressures]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(pressures)))
    bars = ax2.bar(pressures, flows, color=colors, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Backpressure', fontweight='bold')
    ax2.set_ylabel('Mean Flow (μL/min)', fontweight='bold')
    ax2.set_title('Flow vs Tissue Backpressure', fontweight='bold')
    for bar, f in zip(bars, flows):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{f:.1f}', ha='center', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.4)

    # Row 2: Control signals
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(main_df['time'], main_df['I_pump'] * 1000, 'orange', lw=2)
    ax3.fill_between(main_df['time'], main_df['I_pump'] * 1000, alpha=0.3, color='orange')
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Current (mA)', fontweight='bold')
    ax3.set_title('Pump Current', fontweight='bold')
    ax3.grid(True, alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(main_df['time'], main_df['P_bellows'] / 1000, 'brown', lw=2)
    ax4.set_xlabel('Time (s)', fontweight='bold')
    ax4.set_ylabel('Pressure (kPa)', fontweight='bold')
    ax4.set_title('Bellows Pressure', fontweight='bold')
    ax4.grid(True, alpha=0.4)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(main_df['time'], main_df['Glucose'], 'purple', lw=2)
    ax5.axhline(y=8.0, color='red', linestyle='--', label='Threshold')
    ax5.set_xlabel('Time (s)', fontweight='bold')
    ax5.set_ylabel('Glucose (mM)', fontweight='bold')
    ax5.set_title('Blood Glucose', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.4)

    ax6 = fig.add_subplot(gs[1, 3])
    cum = (main_df['Vol_Res'].iloc[0] - main_df['Vol_Res']) * 1e9
    ax6.fill_between(main_df['time'], cum, color='skyblue', alpha=0.7)
    ax6.plot(main_df['time'], cum, 'b-', lw=2)
    ax6.set_xlabel('Time (s)', fontweight='bold')
    ax6.set_ylabel('Volume (nL)', fontweight='bold')
    ax6.set_title('Cumulative Drug Delivery', fontweight='bold')
    ax6.grid(True, alpha=0.4)

    # Row 3: Sensor readings
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(main_df['time'], main_df['V_HF'], 'teal', lw=2)
    ax7.set_xlabel('Time (s)', fontweight='bold')
    ax7.set_ylabel('Voltage (V)', fontweight='bold')
    ax7.set_title('Hot-Film Sensor', fontweight='bold')
    ax7.grid(True, alpha=0.4)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(main_df['time'], main_df['R_RTD'], 'darkred', lw=2)
    ax8.set_xlabel('Time (s)', fontweight='bold')
    ax8.set_ylabel('Resistance (Ω)', fontweight='bold')
    ax8.set_title('RTD Temperature Sensor', fontweight='bold')
    ax8.grid(True, alpha=0.4)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(main_df['time'], main_df['Temp'], 'crimson', lw=2)
    ax9.set_xlabel('Time (s)', fontweight='bold')
    ax9.set_ylabel('Temperature (°C)', fontweight='bold')
    ax9.set_title('Body Temperature', fontweight='bold')
    ax9.grid(True, alpha=0.4)

    ax10 = fig.add_subplot(gs[2, 3])
    ax10.plot(main_df['time'], main_df['Bellows_H'] * 1000, 'darkgreen', lw=2)
    ax10.set_xlabel('Time (s)', fontweight='bold')
    ax10.set_ylabel('Height (mm)', fontweight='bold')
    ax10.set_title('Bellows Expansion', fontweight='bold')
    ax10.grid(True, alpha=0.4)

    # Row 4: Error analysis
    ax11 = fig.add_subplot(gs[3, :2])
    error = (main_df['Q_pred'] - main_df['Q_true']) * to_ul
    ax11.plot(main_df['time'], error, '#22c55e', lw=1.5, alpha=0.9)
    ax11.fill_between(main_df['time'], error, alpha=0.2, color='green')
    ax11.axhline(y=0, color='black', linestyle='--')
    ax11.set_xlabel('Time (s)', fontweight='bold')
    ax11.set_ylabel('Error (μL/min)', fontweight='bold')
    ax11.set_title(f'RF Prediction Error (Mean: {error.mean():.3f}, Std: {error.std():.3f})', fontweight='bold')
    ax11.grid(True, alpha=0.4)

    ax12 = fig.add_subplot(gs[3, 2:])
    tracking_error = (main_df['Q_target'] - main_df['Q_true']) * to_ul
    ax12.plot(main_df['time'], tracking_error, 'darkblue', lw=1.5)
    ax12.fill_between(main_df['time'], tracking_error, alpha=0.3, color='blue')
    ax12.axhline(y=0, color='black', linestyle='--')
    ax12.set_xlabel('Time (s)', fontweight='bold')
    ax12.set_ylabel('Error (μL/min)', fontweight='bold')
    ax12.set_title(f'Control Tracking Error (Mean: {tracking_error.mean():.3f})', fontweight='bold')
    ax12.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_device_structure():
    """Plot 5: All device views in one figure"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Smart Catheter Device Structure - All Views', fontsize=18, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Front View - Relaxed
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_front_view(ax1, 'relaxed')
    ax1.set_title('Front View - Relaxed', fontweight='bold')

    # Front View - Expanded
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_front_view(ax2, 'expanded')
    ax2.set_title('Front View - Expanded', fontweight='bold')

    # Top View
    ax3 = fig.add_subplot(gs[0, 2])
    _draw_top_view(ax3)
    ax3.set_title('Top View', fontweight='bold')

    # Cross Section - Relaxed
    ax4 = fig.add_subplot(gs[1, 0])
    _draw_cross_section(ax4, 'relaxed')
    ax4.set_title('Cross-Section - Relaxed', fontweight='bold')

    # Cross Section - Expanded
    ax5 = fig.add_subplot(gs[1, 1])
    _draw_cross_section(ax5, 'expanded')
    ax5.set_title('Cross-Section - Expanded', fontweight='bold')

    # System Schematic
    ax6 = fig.add_subplot(gs[1, 2])
    _draw_schematic(ax6)
    ax6.set_title('System Schematic', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Helper drawing functions used by plot_device_structure
def _draw_front_view(ax, state):
    width, x_left, x_right = 6, -3, 3
    bellows_h = 8 if state == 'expanded' else 4
    y = 0
    ax.add_patch(Rectangle((x_left - 0.5, y), width + 1, 0.4, facecolor='#555', edgecolor='black', lw=2))
    y += 0.4
    ax.add_patch(Rectangle((x_left, y), width, 0.4, facecolor='#9B59B6', edgecolor='black', lw=2))
    y += 0.4
    water_h = 0.16 if state == 'expanded' else 0.1
    ax.add_patch(Rectangle((x_left, y), width, water_h, facecolor='#6BCF7F', edgecolor='black', lw=1.5, alpha=0.6))
    y += water_h
    bellows_x = [x_left, x_left - 0.5, x_left, x_right, x_right + 0.5, x_right, x_left]
    bellows_y = [y, y + bellows_h * 0.5, y + bellows_h, y + bellows_h, y + bellows_h * 0.5, y, y]
    ax.add_patch(Polygon(list(zip(bellows_x, bellows_y)), facecolor='#FF6B9D', edgecolor='#FF1744', lw=2.5, alpha=0.7))
    y += bellows_h
    ax.add_patch(Rectangle((-0.25, y), 0.5, 0.8, facecolor='#888', edgecolor='black', lw=2))
    ax.set_xlim(x_left - 2, x_right + 2)
    ax.set_ylim(-0.5, y + 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def _draw_top_view(ax):
    ax.add_patch(Circle((0, 0), 3.0, facecolor='#FF6B9D', alpha=0.3, edgecolor='#FF1744', lw=3))
    ax.add_patch(Circle((0, 0), 0.9, facecolor='#888', edgecolor='black', lw=2))
    ax.text(0, 0, 'Port', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def _draw_cross_section(ax, state):
    y_water = 160 if state == 'expanded' else 130
    water_h = 52 if state == 'expanded' else 42
    if state == 'expanded':
        bellows_x = [120, 110, 120, 110, 120, 280, 290, 280, 290, 280, 120]
        bellows_y = [80, 100, 120, 140, 160, 160, 140, 120, 100, 80, 80]
    else:
        bellows_x = [140, 130, 140, 260, 270, 260, 140]
        bellows_y = [100, 115, 130, 130, 115, 100, 100]
    ax.fill(bellows_x, bellows_y, facecolor='#FF6B9D', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(Rectangle((100, y_water + 8), 200, water_h, facecolor='#6BCF7F', edgecolor='black', lw=1, alpha=0.6))
    y_peek = y_water + 8 + water_h + 5
    ax.add_patch(Rectangle((90, y_peek), 220, 30, facecolor='#9B59B6', edgecolor='black', lw=2))
    ax.set_xlim(80, 320)
    ax.set_ylim(70, y_peek + 50)
    ax.grid(True, alpha=0.3)

def _draw_schematic(ax):
    blocks = [
        (0.1, 0.7, 0.25, 0.15, 'Glucose\nSensor', '#E74C3C'),
        (0.1, 0.45, 0.25, 0.15, 'RTD\nSensor', '#3498DB'),
        (0.1, 0.2, 0.25, 0.15, 'Hot-Film\nSensor', '#2ECC71'),
        (0.45, 0.45, 0.2, 0.2, 'Random\nForest', '#22c55e'),
        (0.75, 0.45, 0.2, 0.2, 'PID\nControl', '#F39C12'),
        (0.75, 0.1, 0.2, 0.15, 'Bellows\nPump', '#1ABC9C'),
    ]
    for x, y, w, h, label, color in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', lw=2))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax.annotate('', xy=(0.45, 0.55), xytext=(0.35, 0.77), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.45, 0.55), xytext=(0.35, 0.52), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.45, 0.55), xytext=(0.35, 0.27), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.75, 0.55), xytext=(0.65, 0.55), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.85, 0.25), xytext=(0.85, 0.45), arrowprops=dict(arrowstyle='->', lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def plot_flow_characterization():
    """Plot 6: Flow rate characterization curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Device Characterization Curves', fontsize=16, fontweight='bold')

    currents = np.linspace(0.1e-3, 2.0e-3, 20)
    flows = []
    for I in currents:
        molar_rate = I / (Config.Z_VAL * Config.PHYSICS.faraday_constant)
        raw_vol_rate = molar_rate * Config.PHYSICS.gas_constant * Config.PHYSICS.temperature / Config.P_ATM
        flows.append(raw_vol_rate * 0.8 * Config.TO_UL_MIN)

    axes[0].plot(currents * 1000, flows, 'bo-', lw=2, markersize=8)
    axes[0].fill_between(currents * 1000, flows, alpha=0.3)
    axes[0].set_xlabel('Current (mA)', fontweight='bold')
    axes[0].set_ylabel('Flow Rate (μL/min)', fontweight='bold')
    axes[0].set_title('Flow vs Electrolysis Current', fontweight='bold')
    axes[0].grid(True, alpha=0.4)

    volumes = np.linspace(5e-6, 50e-6, 20)
    pressures = []
    for V in volumes:
        n_moles = (Config.P_ATM * 20e-6) / (Config.PHYSICS.gas_constant * Config.PHYSICS.temperature)
        P = (n_moles * Config.PHYSICS.gas_constant * Config.PHYSICS.temperature) / V
        pressures.append((P - Config.P_ATM) / 1000)

    axes[1].plot(volumes * 1e6, pressures, 'r^-', lw=2, markersize=8)
    axes[1].set_xlabel('Gas Volume (μL)', fontweight='bold')
    axes[1].set_ylabel('Gauge Pressure (kPa)', fontweight='bold')
    axes[1].set_title('Pressure-Volume Relationship', fontweight='bold')
    axes[1].grid(True, alpha=0.4)

    temps = np.linspace(35, 40, 20)
    resistances = Config.R0_RTD * (1 + Config.ALPHA_RTD * (temps - 25))
    axes[2].plot(temps, resistances, 'gs-', lw=2, markersize=8)
    axes[2].set_xlabel('Temperature (°C)', fontweight='bold')
    axes[2].set_ylabel('RTD Resistance (Ω)', fontweight='bold')
    axes[2].set_title('RTD Temperature Response', fontweight='bold')
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_sensor_fusion_analysis(main_df):
    """Plot 7: Sensor fusion deep dive"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensor Fusion Analysis: Why AI Wins', fontsize=16, fontweight='bold')
    to_ul = Config.TO_UL_MIN

    ax = axes[0, 0]
    ax.scatter(main_df['V_HF'], main_df['Q_true'] * to_ul, alpha=0.5, s=20)
    ax.set_xlabel('Hot-Film Voltage (V)', fontweight='bold')
    ax.set_ylabel('True Flow (μL/min)', fontweight='bold')
    ax.set_title('Flow vs Hot-Film Sensor', fontweight='bold')
    ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.scatter(main_df['R_RTD'], main_df['Temp'], alpha=0.5, s=20, c='red')
    ax.set_xlabel('RTD Resistance (Ω)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature vs RTD', fontweight='bold')
    ax.grid(True, alpha=0.4)

    ax = axes[0, 2]
    ax.scatter(main_df['Glucose'], main_df['Q_target'] * to_ul, alpha=0.5, s=20, c='purple')
    ax.set_xlabel('Glucose (mM)', fontweight='bold')
    ax.set_ylabel('Target Flow (μL/min)', fontweight='bold')
    ax.set_title('Target Flow vs Glucose', fontweight='bold')
    ax.grid(True, alpha=0.4)

    ax = axes[1, 0]
    ax.plot(main_df['time'], main_df['Q_true'] * to_ul, 'b-', label='True', lw=2)
    ax.plot(main_df['time'], main_df['Q_Naive'] * to_ul, color='#ef4444', linestyle='--', label='Naive', alpha=0.8)
    ax.plot(main_df['time'], main_df['Q_pred'] * to_ul, color='#22c55e', linestyle=':', label='RF', lw=2.5)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Flow (μL/min)', fontweight='bold')
    ax.set_title('Flow Estimation Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    naive_err = np.abs(main_df['Q_Naive'] - main_df['Q_true']) * to_ul
    ai_err = np.abs(main_df['Q_pred'] - main_df['Q_true']) * to_ul
    ax.plot(main_df['time'], naive_err, color='#ef4444', label='Naive Error', alpha=0.8)
    ax.plot(main_df['time'], ai_err, color='#22c55e', label='RF Error', alpha=0.8)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Absolute Error (μL/min)', fontweight='bold')
    ax.set_title('Estimation Error Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes[1, 2]
    improvement = (naive_err - ai_err)
    ax.hist(improvement, bins=30, color='#22c55e', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', lw=2)
    ax.axvline(x=improvement.mean(), color='black', linestyle='-', lw=2, label=f'Mean: {improvement.mean():.2f}')
    ax.set_xlabel('RF Improvement (μL/min)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Positive Values = RF Wins', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def print_summary_report(models, results_dict):
    """Print comprehensive text summary"""
    to_ul = Config.TO_UL_MIN
    main_df = results_dict['Dynamic']

    print("\n" + "=" * 80)
    print("📊 SMART CATHETER SIMULATION - RESULTS REPORT")
    print("=" * 80)

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                           MODEL PERFORMANCE                                 │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ {'Model':<20} {'R² Score':<15} {'RMSE (μL/min)':<18} {'Improvement':<15} │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")

    naive_rmse = models['Naive']['rmse']
    for name in ['Naive', 'RandomForest']:
        data = models[name]
        imp = (naive_rmse - data['rmse']) / naive_rmse * 100 if name != 'Naive' else 0
        marker = " 🏆" if name == 'RandomForest' else ""
        print(f"│ {name:<20} {data['r2']:<15.4f} {data['rmse']*to_ul:<18.3f} {imp:<14.1f}%{marker} │")

    print("└─────────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                         SIMULATION RESULTS                                  │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Duration:              {Config.DURATION} seconds                                        │")
    print(f"│ Mean Flow Rate:        {main_df['Q_true'].mean() * to_ul:.3f} μL/min                                   │")
    print(f"│ Total Drug Delivered:  {(main_df['Vol_Res'].iloc[0] - main_df['Vol_Res'].iloc[-1]) * 1e9:.1f} nL                                       │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                       CONTROL PERFORMANCE                                   │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    tracking_error = (main_df['Q_target'] - main_df['Q_true']) * to_ul
    pred_error = (main_df['Q_pred'] - main_df['Q_true']) * to_ul
    print(f"│ Tracking Error Mean:   {tracking_error.mean():.4f} μL/min                                  │")
    print(f"│ Prediction Error Mean: {pred_error.mean():.4f} μL/min                                  │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 80)
    print("✅ REPORT COMPLETE")
    print("=" * 80 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("SMART CATHETER - RF vs NAIVE COMPARISON")
    logger.info(f"Training Samples: {Config.TRAINING_SAMPLES:,}")
    logger.info("=" * 80)

    # Train models
    models, X_test, y_test = get_models()
    rf_model = models['RandomForest']['model']

    # Run simulations
    logger.info("\n--- Running Simulations ---")
    results = {'Dynamic': run_simulation_loop(rf_model)}
    for p in [5, 10, 15, 20, 25]:
        results[f'{p} mmHg'] = run_simulation_loop(rf_model, p)

    # Generate all plots
    logger.info("\n--- Generating Visualizations ---")

    logger.info("  [1/7] Direct Model Comparison (Combined Plot)...")
    plot_model_comparison(models, X_test, y_test)

    logger.info("  [2/7] Model Metrics...")
    plot_model_metrics_comparison(models)

    logger.info("  [3/7] Error Distribution...")
    plot_error_distribution(models, y_test)

    logger.info("  [4/7] Comprehensive Dashboard...")
    plot_comprehensive_dashboard(results)

    logger.info("  [5/7] Device Structure...")
    plot_device_structure()

    logger.info("  [6/7] Flow Characterization...")
    plot_flow_characterization()

    logger.info("  [7/7] Sensor Fusion Analysis...")
    plot_sensor_fusion_analysis(results['Dynamic'])

    # Print summary report
    print_summary_report(models, results)

    logger.info("\n✅ ALL VISUALIZATIONS COMPLETE!")

if __name__ == "__main__":
    main()
