/**
 * @file ML related tasks. This code runs in a utility process.
 *
 * The ML runtime we use for inference is [ONNX](https://onnxruntime.ai). Models
 * for various tasks are not shipped with the app but are downloaded on demand.
 */

// See [Note: Using Electron APIs in UtilityProcess] about what we can and
// cannot import.

import Tokenizer from "clip-bpe-js";
import { expose } from "comlink";
import { net } from "electron/main";
import { existsSync } from "fs";
import fs from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { messagePortMainEndpoint } from "../utils/comlink-endpoint";
import { ensure, wait } from "../utils/common";
import { writeStream } from "../utils/stream";

/**
 * We cannot do
 *
 *     import log from "../log";
 *
 * because that requires the Electron APIs that are not available to a utility
 * process (See: [Note: Using Electron APIs in UtilityProcess]). But even if
 * that were to work, logging will still be problematic since we'd try opening
 * the log file from two different Node.js processes (this one, and the main
 * one), and I didn't find any indication in the electron-log repository that
 * the log file's integrity would be maintained in such cases.
 *
 * So instead we create this proxy log object that uses `process.parentPort` to
 * transport the logs over to the main process.
 */
const log = {
    /**
     * Unlike the real {@link log.error}, this accepts only the first string
     * argument, not the second optional error one.
     */
    errorString: (s: string) => mainProcess("log.errorString", s),
    info: (...ms: unknown[]) => mainProcess("log.info", ms),
    /**
     * Unlike the real {@link log.debug}, this is (a) eagerly evaluated, and (b)
     * accepts only strings.
     */
    debugString: (s: string) => mainProcess("log.debugString", s),
};

/**
 * Send a message to the main process using a barebones RPC protocol.
 */
const mainProcess = (method: string, param: unknown) =>
    process.parentPort.postMessage({ method, p: param });

log.debugString(
    `Started ML worker process with args ${process.argv.join(" ")}`,
);

process.parentPort.once("message", (e) => {
    parseInitData(e.data);

    const port = ensure(e.ports[0]);
    expose(
        {
            computeCLIPImageEmbedding,
            computeCLIPTextEmbeddingIfAvailable,
            detectFaces,
            computeFaceEmbeddings,
        },
        messagePortMainEndpoint(port),
    );
    // port.on("message", (request) => {
    //     void handleMessageFromRenderer(request.data).then((response) =>
    //         port.postMessage(response),
    //     );
    // });
    // port.start();
});

/**
 * We cannot access Electron's {@link app} object within a utility process, so
 * we pass the value of `app.getPath("userData")` during initialization, and it
 * can be subsequently retrieved from here.
 */
let _userDataPath: string | undefined;

/** Equivalent to app.getPath("userData") */
const userDataPath = () => ensure(_userDataPath);

const parseInitData = (data: unknown) => {
    if (
        data &&
        typeof data == "object" &&
        "userDataPath" in data &&
        typeof data.userDataPath == "string"
    ) {
        _userDataPath = data.userDataPath;
    } else {
        log.errorString("Unparseable initialization data");
    }
};

/**
 * Our hand-rolled RPC handler and router - the Node.js utility process end.
 *
 * Sibling of the electronMLWorker function (in `ml/worker.ts`) in the web code.
 *
 * [Note: Node.js ML worker RPC protocol]
 *
 * -   Each RPC call (i.e. request message) has a "method" (string), "id"
 *     (number) and "p" (arbitrary param).
 *
 * -   Each RPC result (i.e. response message) has an "id" (number) that is the
 *     same as the "id" for the request which it corresponds to.
 *
 * -   If the RPC call was a success, then the response messege will have an
 *     "result" (arbitrary result) property. Otherwise it will have a "error"
 *     (string) property describing what went wrong.
 */
export const handleMessageFromRenderer = (m: unknown) => {
    if (m && typeof m == "object" && "method" in m && "id" in m && "p" in m) {
        const id = m.id;
        // const p = m.p;
        try {
            switch (m.method) {
                case "foo":
                    // if (p && typeof p == "string")
                    // return { id, result: await foo(p) };
                    break;
            }
        } catch (e) {
            return { id, error: e instanceof Error ? e.message : String(e) };
        }
        return { id, error: "Unknown message" };
    }

    // We don't even have an "id", so at least log it lest the renderer also
    // ignore the "id"-less response.
    log.info("Ignoring unknown message", m);
    return { error: "Unknown message" };
};

/**
 * Return a function that can be used to trigger a download of the specified
 * model, and the creating of an ONNX inference session initialized using it.
 *
 * Multiple parallel calls to the returned function are fine, it ensures that
 * the the model will be downloaded and the session created using it only once.
 * All pending calls to it meanwhile will just await on the same promise.
 *
 * And once the promise is resolved, the create ONNX inference session will be
 * cached, so subsequent calls to the returned function will just reuse the same
 * session.
 *
 * {@link makeCachedInferenceSession} can itself be called anytime, it doesn't
 * actively trigger a download until the returned function is called.
 *
 * @param modelName The name of the model to download.
 *
 * @param modelByteSize The size in bytes that we expect the model to have. If
 * the size of the downloaded model does not match the expected size, then we
 * will redownload it.
 *
 * @returns a function. calling that function returns a promise to an ONNX
 * session.
 */
const makeCachedInferenceSession = (
    modelName: string,
    modelByteSize: number,
) => {
    let session: Promise<ort.InferenceSession> | undefined;

    const download = () =>
        modelPathDownloadingIfNeeded(modelName, modelByteSize);

    const createSession = (modelPath: string) =>
        createInferenceSession(modelPath);

    const cachedInferenceSession = () => {
        if (!session) session = download().then(createSession);
        return session;
    };

    return cachedInferenceSession;
};

/**
 * Download the model named {@link modelName} if we don't already have it.
 *
 * Also verify that the size of the model we get matches {@expectedByteSize} (if
 * not, redownload it).
 *
 * @returns the path to the model on the local machine.
 */
const modelPathDownloadingIfNeeded = async (
    modelName: string,
    expectedByteSize: number,
) => {
    const modelPath = modelSavePath(modelName);

    if (!existsSync(modelPath)) {
        log.info("CLIP image model not found, downloading");
        await downloadModel(modelPath, modelName);
    } else {
        const size = (await fs.stat(modelPath)).size;
        if (size !== expectedByteSize) {
            log.errorString(
                `The size ${size} of model ${modelName} does not match the expected size, downloading again`,
            );
            await downloadModel(modelPath, modelName);
        }
    }

    return modelPath;
};

/** Return the path where the given {@link modelName} is meant to be saved */
const modelSavePath = (modelName: string) =>
    path.join(userDataPath(), "models", modelName);

const downloadModel = async (saveLocation: string, name: string) => {
    // `mkdir -p` the directory where we want to save the model.
    const saveDir = path.dirname(saveLocation);
    await fs.mkdir(saveDir, { recursive: true });
    // Download.
    log.info(`Downloading ML model from ${name}`);
    const url = `https://models.ente.io/${name}`;
    const res = await net.fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
    const body = res.body;
    if (!body) throw new Error(`Received an null response for ${url}`);
    // Save.
    await writeStream(saveLocation, body);
    log.info(`Downloaded CLIP model ${name}`);
};

/**
 * Create an ONNX {@link InferenceSession} with some defaults.
 */
const createInferenceSession = async (modelPath: string) => {
    return await ort.InferenceSession.create(modelPath, {
        // Restrict the number of threads to 1.
        intraOpNumThreads: 1,
        // Be more conservative with RAM usage.
        enableCpuMemArena: false,
    });
};

const cachedCLIPImageSession = makeCachedInferenceSession(
    "clip-image-vit-32-float32.onnx",
    351468764 /* 335.2 MB */,
);

/**
 * Compute CLIP embeddings for an image.
 *
 * The embeddings are computed using ONNX runtime, with CLIP as the model.
 */
export const computeCLIPImageEmbedding = async (input: Float32Array) => {
    const session = await cachedCLIPImageSession();
    const t = Date.now();
    const feeds = {
        input: new ort.Tensor("float32", input, [1, 3, 224, 224]),
    };
    const results = await session.run(feeds);
    log.debugString(`ONNX/CLIP image embedding took ${Date.now() - t} ms`);
    /* Need these model specific casts to type the result */
    return ensure(results.output).data as Float32Array;
};

const cachedCLIPTextSession = makeCachedInferenceSession(
    "clip-text-vit-32-uint8.onnx",
    64173509 /* 61.2 MB */,
);

let _tokenizer: Tokenizer | undefined;
const getTokenizer = () => {
    if (!_tokenizer) _tokenizer = new Tokenizer();
    return _tokenizer;
};

/**
 * Compute CLIP embeddings for an text snippet.
 *
 * The embeddings are computed using ONNX runtime, with CLIP as the model.
 */
export const computeCLIPTextEmbeddingIfAvailable = async (text: string) => {
    const sessionOrSkip = await Promise.race([
        cachedCLIPTextSession(),
        // Wait for a tick to get the session promise to resolved the first time
        // this code runs on each app start (and the model has been downloaded).
        wait(0).then(() => 1),
    ]);

    // Don't wait for the download to complete.
    if (typeof sessionOrSkip == "number") {
        log.info(
            "Ignoring CLIP text embedding request because model download is pending",
        );
        return undefined;
    }

    const session = sessionOrSkip;
    const t = Date.now();
    const tokenizer = getTokenizer();
    const tokenizedText = Int32Array.from(tokenizer.encodeForCLIP(text));
    const feeds = {
        input: new ort.Tensor("int32", tokenizedText, [1, 77]),
    };

    const results = await session.run(feeds);
    log.debugString(`ONNX/CLIP text embedding took ${Date.now() - t} ms`);
    return ensure(results.output).data as Float32Array;
};

const cachedFaceDetectionSession = makeCachedInferenceSession(
    "yolov5s_face_640_640_dynamic.onnx",
    30762872 /* 29.3 MB */,
);

/**
 * Face detection with the YOLO model and ONNX runtime.
 */
export const detectFaces = async (input: Float32Array) => {
    const session = await cachedFaceDetectionSession();
    const t = Date.now();
    const feeds = {
        input: new ort.Tensor("float32", input, [1, 3, 640, 640]),
    };
    const results = await session.run(feeds);
    log.debugString(`ONNX/YOLO face detection took ${Date.now() - t} ms`);
    return ensure(results.output).data;
};

const cachedFaceEmbeddingSession = makeCachedInferenceSession(
    "mobilefacenet_opset15.onnx",
    5286998 /* 5 MB */,
);

/**
 * Face embedding with the MobileFaceNet model and ONNX runtime.
 */
export const computeFaceEmbeddings = async (input: Float32Array) => {
    // Dimension of each face (alias)
    const mobileFaceNetFaceSize = 112;
    // Smaller alias
    const z = mobileFaceNetFaceSize;
    // Size of each face's data in the batch
    const n = Math.round(input.length / (z * z * 3));
    const inputTensor = new ort.Tensor("float32", input, [n, z, z, 3]);

    const session = await cachedFaceEmbeddingSession();
    const t = Date.now();
    const feeds = { img_inputs: inputTensor };
    const results = await session.run(feeds);
    log.debugString(`ONNX/MFNT face embedding took ${Date.now() - t} ms`);
    /* Need these model specific casts to extract and type the result */
    return (results.embeddings as unknown as Record<string, unknown>)
        .cpuData as Float32Array;
};
