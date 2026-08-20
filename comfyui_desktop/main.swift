// ComfyUI Desktop — native macOS shell around a local ComfyUI install.
// Starts the server from ~/ComfyUI, shows the UI in a WKWebView window, and
// shuts the server down on quit. Uses the existing venv, custom nodes and models.
import Cocoa
import WebKit

let COMFY_DIR  = ("~/ComfyUI" as NSString).expandingTildeInPath
let PORT       = 8188
let URL_STRING = "http://127.0.0.1:\(PORT)"
let LOG_PATH   = ("~/Library/Logs/ComfyUI.log" as NSString).expandingTildeInPath

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    var server: Process?
    var status: NSTextField!
    var spinner: NSProgressIndicator!
    var overlay: NSView!

    // MARK: server

    func serverIsUp(_ done: @escaping (Bool) -> Void) {
        var req = URLRequest(url: URL(string: "\(URL_STRING)/system_stats")!)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            done((resp as? HTTPURLResponse)?.statusCode == 200 && data != nil)
        }.resume()
    }

    func startServer() {
        let venvPython = "\(COMFY_DIR)/venv/bin/python"
        guard FileManager.default.isExecutableFile(atPath: venvPython) else {
            fatal("No Python venv found at:\n\(venvPython)\n\nExpected a ComfyUI install at \(COMFY_DIR).")
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: venvPython)
        p.arguments = ["main.py", "--listen", "127.0.0.1", "--port", "\(PORT)"]
        p.currentDirectoryURL = URL(fileURLWithPath: COMFY_DIR)
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "\(COMFY_DIR)/venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        env["VIRTUAL_ENV"] = "\(COMFY_DIR)/venv"
        p.environment = env
        FileManager.default.createFile(atPath: LOG_PATH, contents: nil)
        if let fh = FileHandle(forWritingAtPath: LOG_PATH) {
            p.standardOutput = fh; p.standardError = fh
        }
        do { try p.run(); server = p } catch { fatal("Could not start ComfyUI:\n\(error.localizedDescription)") }
    }

    func waitForServer(attempt: Int = 0) {
        if attempt > 200 { fatal("ComfyUI did not start within ~7 minutes.\n\nSee \(LOG_PATH)"); return }
        if let s = server, !s.isRunning {
            fatal("ComfyUI exited during startup.\n\nSee \(LOG_PATH)"); return
        }
        serverIsUp { up in
            DispatchQueue.main.async {
                if up {
                    self.hideOverlay()
                    self.webView.load(URLRequest(url: URL(string: URL_STRING)!))
                } else {
                    if attempt == 12 { self.status.stringValue = "Loading models — this takes a minute…" }
                    if attempt == 60 { self.status.stringValue = "Still loading (large checkpoints)…" }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.waitForServer(attempt: attempt + 1) }
                }
            }
        }
    }

    // MARK: ui

    func buildOverlay(_ frame: NSRect) -> NSView {
        let v = NSView(frame: frame)
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(calibratedWhite: 0.11, alpha: 1).cgColor
        v.autoresizingMask = [.width, .height]

        spinner = NSProgressIndicator(frame: NSRect(x: frame.width/2 - 16, y: frame.height/2 + 10, width: 32, height: 32))
        spinner.style = .spinning
        spinner.autoresizingMask = [.minXMargin, .maxXMargin, .minYMargin, .maxYMargin]
        spinner.startAnimation(nil)
        v.addSubview(spinner)

        status = NSTextField(labelWithString: "Starting ComfyUI…")
        status.frame = NSRect(x: 0, y: frame.height/2 - 30, width: frame.width, height: 22)
        status.alignment = .center
        status.textColor = NSColor(calibratedWhite: 0.75, alpha: 1)
        status.font = .systemFont(ofSize: 13)
        status.autoresizingMask = [.width, .minYMargin, .maxYMargin]
        v.addSubview(status)
        return v
    }

    func hideOverlay() {
        spinner?.stopAnimation(nil)
        overlay?.removeFromSuperview()
    }

    func fatal(_ msg: String) {
        hideOverlay()
        let a = NSAlert()
        a.messageText = "ComfyUI"
        a.informativeText = msg
        a.addButton(withTitle: "Open Log")
        a.addButton(withTitle: "Quit")
        if a.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.open(URL(fileURLWithPath: LOG_PATH))
        }
        NSApp.terminate(nil)
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        let frame = NSRect(x: 0, y: 0, width: 1440, height: 900)
        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "ComfyUI"
        window.titlebarAppearsTransparent = false
        window.center()
        window.setFrameAutosaveName("ComfyUIWindow")
        window.minSize = NSSize(width: 900, height: 600)

        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .default()          // persist logins/settings between launches
        webView = WKWebView(frame: frame, configuration: cfg)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.allowsMagnification = true
        window.contentView = NSView(frame: frame)
        window.contentView?.addSubview(webView)

        overlay = buildOverlay(frame)
        window.contentView?.addSubview(overlay)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        buildMenu()

        // reuse a server that is already running, otherwise start one
        serverIsUp { up in
            DispatchQueue.main.async {
                if up { self.hideOverlay(); self.webView.load(URLRequest(url: URL(string: URL_STRING)!)) }
                else  { self.startServer(); self.waitForServer() }
            }
        }
    }

    func buildMenu() {
        let main = NSMenu()

        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About ComfyUI", action: #selector(about), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Open Log", action: #selector(openLog), keyEquivalent: "l")
        appMenu.addItem(withTitle: "Open Models Folder", action: #selector(openModels), keyEquivalent: "m")
        appMenu.addItem(withTitle: "Open Output Folder", action: #selector(openOutput), keyEquivalent: "o")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit ComfyUI", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let viewMenu = NSMenu(title: "View")
        viewMenu.addItem(withTitle: "Reload", action: #selector(reload), keyEquivalent: "r")
        viewMenu.addItem(withTitle: "Open in Browser", action: #selector(openBrowser), keyEquivalent: "b")
        viewMenu.addItem(.separator())
        viewMenu.addItem(withTitle: "Actual Size", action: #selector(zoomReset), keyEquivalent: "0")
        viewMenu.addItem(withTitle: "Zoom In",  action: #selector(zoomIn),  keyEquivalent: "+")
        viewMenu.addItem(withTitle: "Zoom Out", action: #selector(zoomOut), keyEquivalent: "-")
        viewItem.submenu = viewMenu

        let editItem = NSMenuItem(); main.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Cut",        action: #selector(NSText.cut(_:)),       keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy",       action: #selector(NSText.copy(_:)),      keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste",      action: #selector(NSText.paste(_:)),     keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu

        NSApp.mainMenu = main
    }

    @objc func about() {
        let a = NSAlert()
        a.messageText = "ComfyUI"
        a.informativeText = "Native shell for the local ComfyUI install at:\n\(COMFY_DIR)\n\nUses its existing venv, custom nodes and models."
        a.runModal()
    }
    @objc func openLog()     { NSWorkspace.shared.open(URL(fileURLWithPath: LOG_PATH)) }
    @objc func openModels()  { NSWorkspace.shared.open(URL(fileURLWithPath: "\(COMFY_DIR)/models")) }
    @objc func openOutput()  { NSWorkspace.shared.open(URL(fileURLWithPath: "\(COMFY_DIR)/output")) }
    @objc func reload()      { webView.reload() }
    @objc func openBrowser() { NSWorkspace.shared.open(URL(string: URL_STRING)!) }
    @objc func zoomReset()   { webView.pageZoom = 1.0 }
    @objc func zoomIn()      { webView.pageZoom += 0.1 }
    @objc func zoomOut()     { webView.pageZoom = max(0.3, webView.pageZoom - 0.1) }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ n: Notification) {
        if let s = server, s.isRunning {
            s.terminate()
            // give it a moment to close cleanly, then make sure it is gone
            let deadline = Date().addingTimeInterval(5)
            while s.isRunning && Date() < deadline { usleep(100_000) }
            if s.isRunning { kill(s.processIdentifier, SIGKILL) }
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
