const { HfInference } = require("@huggingface/inference");
const hf = new HfInference(process.env.HF_KEY);

const text = "Let's use a hugging face AI model";

const convertTextToEmbedding = async (text) => {
    try {
        const result = await hf.featureExtraction({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            inputs: text,
        });
        return result;
    } catch (err) {
        console.error("Error converting text to embeddings:", err);
        throw err;
    }
};

const getEmbeddings = async () => {
    try {
        let embeddings = await convertTextToEmbedding(text);
        console.log(embeddings);
    } catch (err) {
        console.error("Error getting embeddings:", err);
    }
};


getEmbeddings();
