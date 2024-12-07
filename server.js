/** @format */

// Load environment variables
require("dotenv").config();

// Import libraries
const Hapi = require("@hapi/hapi");
const tf = require("@tensorflow/tfjs-node");
const { Firestore } = require("@google-cloud/firestore");
const crypto = require("crypto");

// Custom Error Classes
class ClientError extends Error {
    constructor(message, statusCode = 400) {
        super(message);
        this.statusCode = statusCode;
        this.name = "ClientError";
    }
}

class InputError extends ClientError {
    constructor(message, statusCode = 400) {
        super(message);
        this.name = "InputError";
        this.statusCode = statusCode;
    }
}

// Firestore Service
async function database() {
    const settings = {
        projectId: process.env.PROJECT_ID,
    };
    return new Firestore(process.env.APP_ENV === "local" ? settings : undefined);
}

function modelData(doc) {
    return {
        id: doc.id,
        history: {
            result: doc.data().result,
            createdAt: doc.data().createdAt,
            suggestion: doc.data().suggestion,
            id: doc.id,
        },
    };
}

async function storeData(id, data) {
    const predictCollection = (await database()).collection("predictions");
    return predictCollection.doc(id).set(data);
}

async function getData(id = null) {
    const predictCollection = (await database()).collection("predictions");
    if (id) {
        const doc = await predictCollection.doc(id).get();
        if (!doc.exists) return null;
        return modelData(doc);
    } else {
        const snapshot = await predictCollection.get();
        const allData = [];
        snapshot.forEach(doc => allData.push(modelData(doc)));
        return allData;
    }
}

// Model Service
async function loadModel() {
    const modelPath = process.env.APP_ENV === "local" ? process.env.LOCAL_MODEL_URL : process.env.MODEL_URL;
    console.log(`Trying to load model from: ${modelPath}`);

    try {
        return await tf.loadGraphModel(modelPath);
    } catch (error) {
        console.error("Error loading model:", error);
        throw error;
    }
}

// Inference Service
async function predictClassification(model, image) {
    try {
        if (image.length > 1024 * 1024) throw new InputError("Ukuran gambar terlalu besar. Maksimum 1MB.");

        const tensor = tf.node.decodeJpeg(image).resizeNearestNeighbor([224, 224]).expandDims().toFloat();
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        let result = { confidenceScore, label: "Cancer", suggestion: "Segera periksa ke dokter!" };
        if (confidenceScore < 1) {
            result.label = "Non-cancer";
            result.suggestion = "Penyakit kanker tidak terdeteksi.";
        }

        return result;
    } catch (error) {
        throw new InputError("Terjadi kesalahan dalam melakukan prediksi");
    }
}

// Handlers
async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { model } = request.server.app;

    const { confidenceScore, label, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
        id,
        result: label,
        suggestion,
        confidenceScore,
        createdAt,
    };

    await storeData(id, data);

    const response = h.response({
        status: "success",
        message: confidenceScore > 0 ? "Model is predicted successfully" : "Please use the correct picture",
        data,
    });
    response.code(201);
    return response;
}

async function getPredictHandler(request, h) {
    const { id } = request.params;

    const data = await getData(id);

    if (!data) {
        const response = h.response({
            status: "fail",
            message: "Prediction not found",
        });
        response.code(404);
        return response;
    }

    const response = h.response({
        status: "success",
        data,
    });
    response.code(200);
    return response;
}

// Main Server
(async () => {
    const server = Hapi.server({
        port: process.env.APP_PORT || 3000,
        host: process.env.APP_HOST || "localhost",
        routes: {
            cors: {
                origin: ["*"],
            },
            payload: {
                maxBytes: 1 * 1024 * 1024,
            },
        },
    });

    const model = await loadModel();
    server.app.model = model;

    server.route([
        {
            path: "/predict",
            method: "POST",
            handler: postPredictHandler,
            options: {
                payload: {
                    allow: "multipart/form-data",
                    multipart: true,
                },
            },
        },
        {
            path: "/predict/histories",
            method: "GET",
            handler: getPredictHandler,
            options: {},
        },
    ]);

    server.ext("onPreResponse", (request, h) => {
        const response = request.response;

        if (response.isBoom && response.output.statusCode === 413) {
            const newResponse = h.response({
                status: "fail",
                message: "Payload content length greater than maximum allowed: 1000000",
            });
            newResponse.code(413);
            return newResponse;
        }

        if (response instanceof InputError) {
            const newResponse = h.response({
                status: "fail",
                message: `${response.message}`,
            });
            newResponse.code(response.statusCode);
            return newResponse;
        }

        if (response.isBoom) {
            const newResponse = h.response({
                status: "fail",
                message: response.message,
            });
            newResponse.code(response.output.statusCode);
            return newResponse;
        }

        return h.continue;
    });

    await server.start();
    console.log(`Server start at: ${server.info.uri}`);
})();
