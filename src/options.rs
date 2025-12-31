use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum UciOption {
    Check {
        name: &'static str,
        default: bool,
    },
    Spin {
        name: &'static str,
        default: i64,
        min: i64,
        max: i64,
    },
    String {
        name: &'static str,
        default: &'static str,
    },
}

impl Display for UciOption {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            UciOption::Check { name, default } => {
                write!(f, "option name {name} type check default {default}")
            }
            UciOption::Spin {
                name,
                default,
                min,
                max,
            } => write!(
                f,
                "option name {name} type spin default {default} min {min} max {max}"
            ),
            UciOption::String { name, default } => {
                write!(f, "option name {name} type string default {default}")
            }
        }
    }
}

impl UciOption {
    const fn check(name: &'static str, default: bool) -> Self {
        UciOption::Check { name, default }
    }

    const fn spin(name: &'static str, default: i64, min: i64, max: i64) -> Self {
        UciOption::Spin {
            name,
            default,
            min,
            max,
        }
    }

    const fn string(name: &'static str, default: &'static str) -> Self {
        UciOption::String { name, default }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            UciOption::Check { name, .. }
            | UciOption::Spin { name, .. }
            | UciOption::String { name, .. } => name,
        }
    }

    #[must_use]
    pub fn default(&self) -> String {
        match self {
            UciOption::Check { default, .. } => default.to_string(),
            UciOption::Spin { default, .. } => default.to_string(),
            UciOption::String { default, .. } => (*default).to_string(),
        }
    }

    pub fn print_all() {
        for option in ALL_OPTIONS {
            println!("{option}");
        }
    }
}

static HASH: UciOption = UciOption::spin("Hash", 128, 1, 2 << 24);
static THREADS: UciOption = UciOption::spin("Threads", 1, 1, 1 << 16);
static MULTI_PV: UciOption = UciOption::spin("MultiPV", 1, 1, 255);
static SYZYGY_PATH: UciOption = UciOption::string("SyzygyPath", "<empty>");

static CPUCT: UciOption = UciOption::spin("CPuct", 16, 1, 2 << 16);
static CPUCT_TAU: UciOption = UciOption::spin("CPuctTau", 84, 0, 100);
static CPUCT_JITTER: UciOption = UciOption::spin("CPuctJitter", 5, 0, 100);
static CPUCT_TREND_ADJUSTMENT: UciOption = UciOption::spin("CPuctTrendAdjustment", 15, -100, 100);
static CPUCT_GINI_BASE: UciOption = UciOption::spin("CPuctGiniBase", 68, 0, 2 << 16);
static CPUCT_GINI_FACTOR: UciOption = UciOption::spin("CPuctGiniFactor", 163, 0, 2 << 16);
static CPUCT_GINI_MAX: UciOption = UciOption::spin("CPuctGiniMax", 210, 0, 2 << 16);
static CVISITS_SELECTION: UciOption = UciOption::spin("CVisitsSelection", 1, 0, 100);
static POLICY_TEMPERATURE: UciOption = UciOption::spin("PolicyTemperature", 100, 0, 2 << 16);
static POLICY_TEMPERATURE_ROOT: UciOption =
    UciOption::spin("PolicyTemperatureRoot", 1450, 0, 2 << 16);

static TM_MIN_M: UciOption = UciOption::spin("TMMinM", 10, 0, 2 << 16);
static TM_MAX_M: UciOption = UciOption::spin("TMMaxM", 500, 0, 2 << 16);
static TM_VISITS_BASE: UciOption = UciOption::spin("TMVisitsBase", 140, 0, 2 << 16);
static TM_VISITS_M: UciOption = UciOption::spin("TMVisitsM", 139, 0, 2 << 16);
static TM_PV_DIFF_C: UciOption = UciOption::spin("TMPvDiffC", 0, 0, 100);
static TM_PV_DIFF_M: UciOption = UciOption::spin("TMPvDiffM", 121, 0, 2 << 16);

static ENABLE_MATERIAL_SCALING: UciOption = UciOption::check("EnableMaterialScaling", true);
static ENABLE_50MR_SCALING: UciOption = UciOption::check("Enable50mrScaling", true);

static CHESS960: UciOption = UciOption::check("UCI_Chess960", false);
static POLICY_ONLY: UciOption = UciOption::check("PolicyOnly", false);
static SHOW_MOVESLEFT: UciOption = UciOption::check("UCI_ShowMovesLeft", false);
static SHOW_WDL: UciOption = UciOption::check("UCI_ShowWDL", false);

static ALL_OPTIONS: &[UciOption] = &[
    HASH,
    THREADS,
    MULTI_PV,
    SYZYGY_PATH,
    CPUCT,
    CPUCT_TAU,
    CPUCT_JITTER,
    CPUCT_TREND_ADJUSTMENT,
    CPUCT_GINI_BASE,
    CPUCT_GINI_FACTOR,
    CPUCT_GINI_MAX,
    CVISITS_SELECTION,
    POLICY_TEMPERATURE,
    POLICY_TEMPERATURE_ROOT,
    TM_MIN_M,
    TM_MAX_M,
    TM_VISITS_BASE,
    TM_VISITS_M,
    TM_PV_DIFF_C,
    TM_PV_DIFF_M,
    ENABLE_MATERIAL_SCALING,
    ENABLE_50MR_SCALING,
    CHESS960,
    POLICY_ONLY,
    SHOW_MOVESLEFT,
    SHOW_WDL,
];

pub struct UciOptionMap {
    inner: HashMap<UciOption, String>,
}

impl Default for UciOptionMap {
    fn default() -> Self {
        let mut inner = HashMap::new();

        for option in ALL_OPTIONS {
            inner.insert(*option, option.default());
        }

        Self { inner }
    }
}

impl UciOptionMap {
    #[must_use]
    pub fn get(&self, option: &UciOption) -> Option<&str> {
        self.inner.get(option).map(String::as_str)
    }

    /// Gets an option value and applies a parsing function, falling back to default.
    ///
    /// # Panics
    ///
    /// Panics if the parsing function fails on the option's default value.
    pub fn get_and<F, T>(&self, option: &UciOption, f: F) -> T
    where
        F: FnOnce(&str) -> Option<T> + Copy,
    {
        if let Some(value) = self.get(option).and_then(f) {
            return value;
        }

        println!(
            "info string Invalid value for option '{}'. Using default.",
            option.name()
        );
        f(&option.default()).unwrap()
    }

    #[must_use]
    pub fn get_f32(&self, option: &UciOption) -> f32 {
        self.get_and(option, |s| s.parse().map(|f: f32| f / 100.).ok())
    }

    pub fn set(&mut self, name: &str, value: &str) {
        for option in ALL_OPTIONS {
            if option.name().eq_ignore_ascii_case(name) {
                self.inner.insert(*option, value.to_owned());
                return;
            }
        }

        println!("info string Unknown option '{name}'");
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy)]
pub struct EvaluationOptions {
    pub enable_material_scaling: bool,
    pub enable_50mr_scaling: bool,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy)]
pub struct MctsOptions {
    pub cpuct: f32,
    pub cpuct_tau: f32,
    pub cpuct_jitter: f32,
    pub cpuct_trend_adjustment: f32,
    pub cpuct_gini_base: f32,
    pub cpuct_gini_factor: f32,
    pub cpuct_gini_max: f32,
    pub policy_temperature: f32,
    pub policy_temperature_root: f32,
}

#[allow(clippy::module_name_repetitions, clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
pub struct EngineOptions {
    pub hash_size_mb: usize,
    pub threads: u32,
    pub c_visits_selection: f32,
    pub multi_pv: usize,
    pub is_chess960: bool,
    pub is_policy_only: bool,
    pub show_movesleft: bool,
    pub show_wdl: bool,
    pub mcts_options: MctsOptions,
    pub time_management_options: TimeManagementOptions,
    pub evaluation_options: EvaluationOptions,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy)]
pub struct TimeManagementOptions {
    pub min_m: f32,
    pub max_m: f32,
    pub visits_base: f32,
    pub visits_m: f32,
    pub pv_diff_c: f32,
    pub pv_diff_m: f32,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        EvaluationOptions::from(&UciOptionMap::default())
    }
}

impl From<&UciOptionMap> for EvaluationOptions {
    fn from(map: &UciOptionMap) -> Self {
        Self {
            enable_material_scaling: map.get_and(&ENABLE_MATERIAL_SCALING, |s| s.parse().ok()),
            enable_50mr_scaling: map.get_and(&ENABLE_50MR_SCALING, |s| s.parse().ok()),
        }
    }
}

impl Default for MctsOptions {
    fn default() -> Self {
        MctsOptions::from(&UciOptionMap::default())
    }
}

impl From<&UciOptionMap> for MctsOptions {
    fn from(map: &UciOptionMap) -> Self {
        Self {
            cpuct: map.get_f32(&CPUCT),
            cpuct_tau: map.get_f32(&CPUCT_TAU),
            cpuct_jitter: map.get_f32(&CPUCT_JITTER),
            cpuct_trend_adjustment: map.get_f32(&CPUCT_TREND_ADJUSTMENT),
            cpuct_gini_base: map.get_f32(&CPUCT_GINI_BASE),
            cpuct_gini_factor: map.get_f32(&CPUCT_GINI_FACTOR),
            cpuct_gini_max: map.get_f32(&CPUCT_GINI_MAX),
            policy_temperature: map.get_f32(&POLICY_TEMPERATURE),
            policy_temperature_root: map.get_f32(&POLICY_TEMPERATURE_ROOT),
        }
    }
}

impl From<&UciOptionMap> for TimeManagementOptions {
    fn from(map: &UciOptionMap) -> Self {
        Self {
            min_m: map.get_f32(&TM_MIN_M),
            max_m: map.get_f32(&TM_MAX_M),
            visits_base: map.get_f32(&TM_VISITS_BASE),
            visits_m: map.get_f32(&TM_VISITS_M),
            pv_diff_c: map.get_f32(&TM_PV_DIFF_C),
            pv_diff_m: map.get_f32(&TM_PV_DIFF_M),
        }
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions::from(&UciOptionMap::default())
    }
}

impl From<&UciOptionMap> for EngineOptions {
    fn from(map: &UciOptionMap) -> Self {
        Self {
            hash_size_mb: map.get_and(&HASH, |s| s.parse().ok()),
            threads: map.get_and(&THREADS, |s| s.parse().ok()),
            c_visits_selection: map.get_f32(&CVISITS_SELECTION),
            multi_pv: map.get_and(&MULTI_PV, |s| s.parse().ok()),
            is_chess960: map.get_and(&CHESS960, |s| s.parse().ok()),
            is_policy_only: map.get_and(&POLICY_ONLY, |s| s.parse().ok()),
            show_movesleft: map.get_and(&SHOW_MOVESLEFT, |s| s.parse().ok()),
            show_wdl: map.get_and(&SHOW_WDL, |s| s.parse().ok()),
            mcts_options: MctsOptions::from(map),
            time_management_options: TimeManagementOptions::from(map),
            evaluation_options: EvaluationOptions::from(map),
        }
    }
}
