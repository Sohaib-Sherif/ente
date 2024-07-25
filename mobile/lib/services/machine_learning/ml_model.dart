import "dart:io" show File, Platform;

import "package:logging/logging.dart";
import "package:onnxruntime/onnxruntime.dart";
import "package:photos/services/machine_learning/onnx_env.dart";
import "package:photos/services/remote_assets_service.dart";

abstract class MlModel {
  static final Logger isolateLogger = Logger("MlModelInIsolate");
  Logger get logger;

  String get kModelBucketEndpoint => "https://models.ente.io/";

  String get modelRemotePath;

  String get modelName;

  static final bool _usePlatformPlugin = Platform.isAndroid;

  bool get isInitialized =>
      _usePlatformPlugin ? _isNativePluginInitialized : _isFfiInitialized;
  int get sessionAddress =>
      _usePlatformPlugin ? _nativePluginSessionIndex : _ffiSessionAddress;

  // isInitialized is used to check if the model is loaded by the ffi based
  // plugin
  bool _isFfiInitialized = false;
  int _ffiSessionAddress = -1;

  bool _isNativePluginInitialized = false;
  int _nativePluginSessionIndex = -1;

  Future<(String, String)> getModelNameAndPath() async {
    final path =
        await RemoteAssetsService.instance.getAssetPath(modelRemotePath);
    return (modelName, path);
  }

  void storeSessionAddress(int address) {
    if (_usePlatformPlugin) {
      _nativePluginSessionIndex = address;
      _isNativePluginInitialized = true;
    } else {
      _ffiSessionAddress = address;
      _isFfiInitialized = true;
    }
  }

  void releaseSessionAddress() {
    if (_usePlatformPlugin) {
      _nativePluginSessionIndex = -1;
      _isNativePluginInitialized = false;
    } else {
      _ffiSessionAddress = -1;
      _isFfiInitialized = false;
    }
  }

  // Initializes the model.
  // If `useEntePlugin` is set to true, the custom plugin is used for initialization.
  // Note: The custom plugin requires a dedicated isolate for loading the model to ensure thread safety and performance isolation.
  // In contrast, the current FFI-based plugin leverages the session memory address for session management, which does not require a dedicated isolate.
  static Future<int> loadModel(
    String modelName,
    String modelPath,
  ) async {
    if (_usePlatformPlugin) {
      return await _loadModelWithPlatformPlugin(modelName, modelPath);
    } else {
      return await _loadModelWithFFI(modelName, modelPath);
    }
  }

  static Future<int> _loadModelWithPlatformPlugin(
    String modelName,
    String modelPath,
  ) async {
    final startTime = DateTime.now();
    isolateLogger.info('Initializing $modelName with EntePlugin');
    final OnnxDart plugin = OnnxDart();
    final bool? initResult = await plugin.init(modelName, modelPath);
    if (initResult == null || !initResult) {
      isolateLogger.severe("Failed to initialize $modelName with EntePlugin.");
      throw Exception("Failed to initialize $modelName with EntePlugin.");
    }
    final endTime = DateTime.now();
    isolateLogger.info(
      "$modelName loaded via EntePlugin in ${endTime.difference(startTime).inMilliseconds}ms",
    );
    return 0;
  }

  static Future<int> _loadModelWithFFI(
    String modelName,
    String modelPath,
  ) async {
    isolateLogger.info('Initializing $modelName with FFI');
    ONNXEnvFFI.instance.initONNX(modelName);
    try {
      final startTime = DateTime.now();
      final sessionOptions = OrtSessionOptions()
        ..setInterOpNumThreads(1)
        ..setIntraOpNumThreads(1)
        ..setSessionGraphOptimizationLevel(
          GraphOptimizationLevel.ortEnableAll,
        );
      final session = OrtSession.fromFile(File(modelPath), sessionOptions);
      final endTime = DateTime.now();
      isolateLogger.info(
        "$modelName loaded with FFI, took: ${endTime.difference(startTime).inMilliseconds}ms",
      );
      return session.address;
    } catch (e) {
      rethrow;
    }
  }

  static Future<void> releaseModel(String modelName, int sessionAddress) async {
    if (_usePlatformPlugin) {
      await _releaseModelWithPlatformPlugin(modelName);
    } else {
      await _releaseModelWithFFI(modelName, sessionAddress);
    }
  }

  static Future<void> _releaseModelWithPlatformPlugin(String modelName) async {
    final OnnxDart plugin = OnnxDart();
    final bool? initResult = await plugin.release(modelName);
    if (initResult == null || !initResult) {
      isolateLogger.severe("Failed to release $modelName with PlatformPlugin.");
      throw Exception("Failed to release $modelName with PlatformPlugin.");
    }
  }

  static Future<void> _releaseModelWithFFI(
    String modelName,
    int sessionAddress,
  ) async {
    if (sessionAddress == 0 || sessionAddress == -1) {
      return;
    }
    final session = OrtSession.fromAddress(sessionAddress);
    session.release();
    ONNXEnvFFI.instance.releaseONNX(modelName);
    return;
  }
}
