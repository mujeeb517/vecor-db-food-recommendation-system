const ChromaClient = require("chromadb").ChromaClient;
const { HfInference } = require("@huggingface/inference");
const foodItems = require('./data/FoodDataSet.json');

const hf = new HfInference(process.env.HF_KEY);
const client = new ChromaClient({ path: "http://localhost:8000" });

async function generateEmbeddings(texts) {
    const results = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: texts,
    });
    return results;
}

async function feedData() {
    const uniqueIds = new Set();
    foodItems.forEach((food, index) => {
        while (uniqueIds.has(food.food_id.toString())) {
            food.food_id = `${food.food_id}_${index}`;
        }
        uniqueIds.add(food.food_id.toString());
    });

    const foodTexts = foodItems.map((food) => `${food.food_name}. ${food.food_description}. Ingredients: ${food.food_ingredients.join(", ")}`);
    const embeddingsData = await generateEmbeddings(foodTexts);

    const ids = foodItems.map((food) => food.food_id.toString());

    const collectionName = "food_collection";
    const collection = await client.getOrCreateCollection({
        name: collectionName
    });

    await collection.add({
        ids: ids,
        documents: foodTexts,
        embeddings: embeddingsData,
    });
}

module.exports = { feedData };
