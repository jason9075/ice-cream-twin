//! Deliberate allowlist: view settings and section openness, never simulation parameters.
use icecream_view::ViewOptions;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub view: ViewOptions,
    pub sections: BTreeMap<String, bool>,
}

pub struct Store {
    path: Option<PathBuf>,
    saved: serde_json::Value,
}

impl Store {
    pub fn load() -> (Self, Preferences) {
        let base = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()).map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")));
        Self::load_path(base.map(|p| p.join("ice-cream-sim/ui.json")))
    }

    fn load_path(path: Option<PathBuf>) -> (Self, Preferences) {
        let prefs = path.as_ref().and_then(|p| {
            match std::fs::read(p) {
                Ok(bytes) => match serde_json::from_slice::<Preferences>(&bytes) {
                    Ok(mut prefs) => {
                        // Only the sliders' supported ranges are valid persisted values.
                        prefs.view.surface_radius = prefs.view.surface_radius.clamp(0.5, 3.0);
                        prefs.view.surface_kernel = prefs.view.surface_kernel.clamp(1.5, 6.0);
                        prefs.view.surface_voxel = prefs.view.surface_voxel.clamp(0.3, 2.0);
                        prefs.view.surface_smooth = prefs.view.surface_smooth.clamp(0.0, 4.0);
                        Some(prefs)
                    }
                    Err(e) => { eprintln!("UI preferences {}: {e}", p.display()); None }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => { eprintln!("UI preferences {}: {e}", p.display()); None }
            }
        }).unwrap_or_default();
        let saved = serde_json::to_value(&prefs).expect("UI preferences");
        (Self { path, saved }, prefs)
    }

    pub fn save(&mut self, view: &ViewOptions, sections: &BTreeMap<String, bool>) {
        let prefs = Preferences { view: view.clone(), sections: sections.clone() };
        let value = serde_json::to_value(&prefs).expect("UI preferences");
        if value == self.saved { return; }
        self.saved = value;
        let Some(path) = &self.path else { return };
        let result = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let temp = path.with_extension(format!("{}.tmp", std::process::id()));
            std::fs::write(&temp, serde_json::to_vec_pretty(&prefs)?)?;
            std::fs::rename(temp, path)
        })();
        if let Err(e) = result { eprintln!("Save UI preferences {}: {e}", path.display()); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_saves_only_view_and_sections() {
        let dir = std::env::temp_dir().join(format!("icecream-prefs-{}", std::process::id()));
        let path = dir.join("ui.json");
        let (mut store, mut prefs) = Store::load_path(Some(path.clone()));
        prefs.view.grid_boxes = true;
        prefs.view.surface_radius = 2.0;
        prefs.sections.insert("View".into(), true);
        prefs.sections.insert("Robot arm (SO-101)".into(), false);
        store.save(&prefs.view, &prefs.sections);
        let bytes = std::fs::read(&path).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 2);
        assert!(value.get("view").is_some() && value.get("sections").is_some());
        let (_, loaded) = Store::load_path(Some(path.clone()));
        assert!(loaded.view.grid_boxes);
        assert_eq!(loaded.view.surface_radius, 2.0);
        assert_eq!(loaded.sections, prefs.sections);
        // Unknown simulation fields must never become persisted preferences.
        std::fs::write(&path, br#"{"fluid":{"gravity":0},"view":{"grid_boxes":true}}"#).unwrap();
        let (_, loaded) = Store::load_path(Some(path.clone()));
        assert!(loaded.view.grid_boxes);
        assert_eq!(loaded.view.surface_radius, ViewOptions::default().surface_radius);
        assert!(serde_json::to_value(loaded).unwrap().get("fluid").is_none());
        std::fs::write(&path, b"{broken").unwrap();
        let (_, loaded) = Store::load_path(Some(path));
        assert!(!loaded.view.grid_boxes);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
