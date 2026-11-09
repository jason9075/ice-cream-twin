//! Append-only operation journal. No buffered writer: completed lines survive process termination.
use serde_json::{json, Value};
use std::{fs::{File, OpenOptions}, io::Write, path::PathBuf, sync::{Mutex, OnceLock}, time::{Instant, SystemTime, UNIX_EPOCH}};

#[derive(Default)]
struct Journal {
    active: Option<(File, Instant)>,
    path: Option<PathBuf>,
    error: Option<String>,
}
static JOURNAL: OnceLock<Mutex<Journal>> = OnceLock::new();
fn journal() -> &'static Mutex<Journal> { JOURNAL.get_or_init(|| Mutex::new(Journal::default())) }

pub fn active() -> bool { journal().lock().unwrap().active.is_some() }
pub fn path() -> Option<PathBuf> { journal().lock().unwrap().path.clone() }
pub fn status() -> String {
    let j = journal().lock().unwrap();
    if let Some(e) = &j.error { return format!("Recording error: {e}"); }
    j.path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
}
pub fn start(initial: Value) {
    let result = (|| -> std::io::Result<(File, PathBuf)> {
        let dir = std::env::current_dir()?.join("recordings");
        std::fs::create_dir_all(&dir)?;
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let path = dir.join(format!("operations-{stamp}-{}.jsonl", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).open(&path)?;
        Ok((file, path))
    })();
    let mut j = journal().lock().unwrap();
    match result {
        Ok((file, path)) => { j.active = Some((file, Instant::now())); j.path = Some(path); j.error = None; }
        Err(e) => { j.error = Some(e.to_string()); return; }
    }
    drop(j);
    event("start", json!({"version":1, "unixMs":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(), "initial":initial}));
}
pub fn event(kind: &str, data: Value) {
    let mut j = journal().lock().unwrap();
    let Some((file, start)) = &mut j.active else { return };
    // Serialize before writing so a serialization failure cannot corrupt an earlier line.
    let result = (|| -> std::io::Result<()> {
        let mut line = serde_json::to_vec(&json!({"elapsedMs":start.elapsed().as_secs_f64()*1000.0,"kind":kind,"data":data}))?;
        line.push(b'\n');
        file.write_all(&line)
    })();
    if let Err(e) = result { j.error = Some(e.to_string()); j.active = None; }
}
pub fn stop() {
    event("stop", Value::Null);
    let mut j = journal().lock().unwrap();
    if let Some((file, _)) = j.active.take() {
        if let Err(e) = file.sync_data() { j.error = Some(e.to_string()); }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn survives_process_exit() {
        const CHILD: &str = "ICECREAM_JOURNAL_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            super::start(serde_json::json!({"test":true}));
            super::event("command", serde_json::json!({"type":"eat"}));
            // Exit without stop or destructors, as on abrupt application termination.
            std::process::exit(73);
        }
        let dir = std::env::temp_dir().join(format!("icecream-journal-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "recording::tests::survives_process_exit", "--nocapture"])
            .env(CHILD, "1").current_dir(&dir).status().unwrap();
        assert_eq!(status.code(), Some(73));
        let file = std::fs::read_dir(dir.join("recordings")).unwrap().next().unwrap().unwrap().path();
        let text = std::fs::read_to_string(file).unwrap();
        let lines: Vec<serde_json::Value> = text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["kind"], "start");
        assert_eq!(lines[1]["data"]["type"], "eat");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
