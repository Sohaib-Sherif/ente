import "dart:io";
import "dart:typed_data";

import "package:logging/logging.dart";
import "package:onnxruntime/onnxruntime.dart";
import "package:photos/services/machine_learning/ml_model.dart";
import "package:photos/utils/image_ml_util.dart";
import "package:photos/utils/ml_util.dart";

class ClipImageEncoder extends MlModel {
  static const kRemoteBucketModelPath = "clip-image-vit-32-float32.onnx";

  @override
  String get modelRemotePath => kModelBucketEndpoint + kRemoteBucketModelPath;

  @override
  Logger get logger => _logger;
  static final _logger = Logger('ClipImageEncoder');

  // Singleton pattern
  ClipImageEncoder._privateConstructor();
  static final instance = ClipImageEncoder._privateConstructor();
  factory ClipImageEncoder() => instance;

  static Future<List<double>> predict(Map args) async {
    final imageData = await File(args["imagePath"]).readAsBytes();
    final image = await decodeImageFromData(imageData);
    final ByteData imgByteData = await getByteDataFromImage(image);

    final inputList = await preprocessImageClip(image, imgByteData);

    final inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList, [1, 3, 224, 224]);
    final inputs = {'input': inputOrt};
    final session = OrtSession.fromAddress(args["address"]);
    final runOptions = OrtRunOptions();
    final outputs = session.run(runOptions, inputs);
    final embedding = (outputs[0]?.value as List<List<double>>)[0];

    normalizeEmbedding(embedding);

    return embedding;
  }
}
