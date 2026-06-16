import 'package:flutter/foundation.dart'
    show debugPrint, kDebugMode, kIsWeb, defaultTargetPlatform, TargetPlatform;

/// Base URL for the NeuroScanAi FastAPI backend (`backend/main.py`).
///
/// Override at build/run time, e.g.:
/// `flutter run --dart-define=NEUROSCAN_API_URL=http://192.168.1.5:8000`
///
/// **Flutter Web (Chrome):** if you pass a LAN IP (`192.168.x.x`, `10.x`, etc.), it is
/// automatically rewritten to `http://127.0.0.1:<port>` so Chrome can reach an API on the
/// same PC (Private Network Access blocks `localhost` → LAN otherwise).
/// To force the LAN URL on web (API on another machine), use:
/// `--dart-define=NEUROSCAN_WEB_USE_LAN=true`
class NeuroscanApiConfig {
  NeuroscanApiConfig._();

  static var _loggedResolvedBase = false;

  static const String _dartDefine =
      String.fromEnvironment('NEUROSCAN_API_URL', defaultValue: '');

  /// Set `true` only if the API runs on another device and you use Flutter **web**.
  static const bool _webUseLan =
      bool.fromEnvironment('NEUROSCAN_WEB_USE_LAN', defaultValue: false);

  /// True when web should call the LAN URL from `NEUROSCAN_API_URL` (API not on this PC).
  static bool get webUseLan => _webUseLan;

  /// For physical Android device (USB), use your PC LAN IP.
  /// (If you rebuild APKs after changing PC IP, update this value.)
  static String get baseUrl {
    final String raw;
    if (_dartDefine.isNotEmpty) {
      raw = _dartDefine;
    } else if (kIsWeb) {
      raw = 'http://127.0.0.1:8000';
    } else if (defaultTargetPlatform == TargetPlatform.android) {
      raw = 'http://172.20.10.2:8000';
    } else {
      raw = 'http://127.0.0.1:8000';
    }

    var url = raw.replaceAll(RegExp(r'/$'), '');
    final uri = Uri.tryParse(url);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) return url;

    // Flutter web is served from localhost; Chrome Private Network Access often blocks
    // localhost ->192.168.x.x. When the API is on the same machine, use 127.0.0.1.
    // If the API is on another device, set: --dart-define=NEUROSCAN_WEB_USE_LAN=true
    if (kIsWeb && !_webUseLan && _isPrivateLanHost(uri.host)) {
      url = uri.replace(host: '127.0.0.1').toString();
    }
    // baseUrl is read very often; log once to avoid flooding the console.
    if (kDebugMode && !_loggedResolvedBase) {
      _loggedResolvedBase = true;
      final define = _dartDefine.isEmpty ? '(default)' : _dartDefine;
      debugPrint(
        'NeuroscanApi → $url | NEUROSCAN_API_URL=$define | webUseLan=$_webUseLan',
      );
    }
    return url;
  }

  static bool _isPrivateLanHost(String host) {
    if (host == 'localhost' || host == '127.0.0.1') return false;
    final parts = host.split('.');
    if (parts.length != 4) return false;
    final a = int.tryParse(parts[0]);
    final b = int.tryParse(parts[1]);
    if (a == null || b == null) return false;
    if (a == 10) return true;
    if (a == 192 && b == 168) return true;
    if (a == 172 && b >= 16 && b <= 31) return true;
    return false;
  }
}
