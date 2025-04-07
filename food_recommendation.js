const ChromaClient = require("chromadb").ChromaClient;
const { HfInference } = require("@huggingface/inference");
const { performSimilaritySearch } = require("./similarity_search");

const hf = new HfInference(process.env.HF_KEY);
const client = new ChromaClient({ path: "http://localhost:8000" });


async function classifyText(text, labels) {
    const response = await hf.request({
        model: "facebook/bart-large-mnli",
        inputs: text,
        parameters: { candidate_labels: labels },
    });
    return response;
}

async function extractFilterCriteria(query) {
    const criteria = { diet: null, cuisine: null };

    const dietLabels = ["vegan", "non-vegan", "vegetarian", "non-vegetarian", "pescatarian", "omnivore", "paleo", "ketogenic"];
    const cuisineLabels = ["chinese", "indian", "japanese"];

    const dietResult = await classifyText(query, dietLabels);
    const highestDietScoreLabel = dietResult.labels[0];
    const dietScore = dietResult.scores[0];

    // Only apply diet criteria if the score is very high (e.g., > 0.8)
    if (dietScore > 0.8) {
        criteria.diet = highestDietScoreLabel;
    } else {
        const cuisineResult = await classifyText(query, cuisineLabels);
        const highestCuisineScoreLabel = cuisineResult.labels[0];
        const cuisineScore = cuisineResult.scores[0];

        // Only apply cuisine criteria if the score is very high (e.g., > 0.8)
        if (cuisineScore > 0.8) {
            criteria.cuisine = highestCuisineScoreLabel;
        }
    }
    console.log('Extracted Filter Criteria:', criteria);
    return criteria;
}

(async function main() {
    try {
        // feedData();
        const collectionName = "food_collection";
        const collection = await client.getOrCreateCollection({
            name: collectionName
        });
        const query = "I want to eat vegan food";
        const filterCriteria = await extractFilterCriteria(query);
        const initialResults = await performSimilaritySearch(collection, query, filterCriteria);
        initialResults.slice(0, 5).forEach((item, index) => {
            console.log(`Top ${index + 1} Recommended Food Name ==>, ${item.food_name}`);
        });
    } catch (error) {
        console.error("Error:", error);
    }
})();

