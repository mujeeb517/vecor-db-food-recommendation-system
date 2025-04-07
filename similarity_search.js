const ChromaClient = require("chromadb").ChromaClient;
const { HfInference } = require("@huggingface/inference");
const foodItems = require('./data/FoodDataSet.json');

const hf = new HfInference(process.env.HF_KEY);

const collectionName = "food_collection";

async function generateEmbeddings(texts) {
    const results = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: texts,
    });
    return results;
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);
        console.log(filterCriteria);
        const results = await collection.query({
            collection: collectionName,
            queryEmbeddings: queryEmbedding,
            n: 5,
        });

        if (!results || results.length === 0) {
            console.log(`No food items found similar to "${queryTerm}"`);
            return [];
        }

        let topFoodItems = results.ids[0].map((id, index) => {
            return {
                id,
                score: results.distances[0][index],
                food_name: foodItems.find(item => item.food_id.toString() === id).food_name,
                food_description: foodItems.find(item => item.food_id.toString() === id).food_description
            };
        }).filter(Boolean);
        return topFoodItems.sort((a, b) => a.score - b.score);
    } catch (error) {
        console.error("Error during similarity search:", error);
        return [];
    }
}


module.exports = { performSimilaritySearch };