fn main() {
    // Embed the nine-square mark as the Windows executable icon so pssa.exe
    // shows it in Explorer, the taskbar and the console window. The resource
    // compile only runs for Windows targets; other builds are unaffected.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/pssa.ico");
        res.compile().expect("failed to embed the Windows icon resource");
    }
}
