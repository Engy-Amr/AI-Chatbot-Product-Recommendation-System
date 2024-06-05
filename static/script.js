// Import jQuery using ES6 module syntax
import $ from 'jquery'; // Import jQuery using ES6 module syntax
import { load as spacyLoad } from 'spacy'; // Import specific function from spacy
import * as natural from 'natural'; // Import entire natural module

// TensorFlow.js library can be included via CDN link in HTML
// No equivalent for 'canvas', consider alternative browser APIs

// Function to make AJAX request to Flask app
function getRecommendations(userId) {
    // Make AJAX POST request to /recommend endpoint
    $.ajax({
        url: '/recommend',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ user_id: userId }),
        success: function(response) {
            // Handle successful response
            console.log('Recommendations:', response); // Log recommendations to console
            displayRecommendations(response);
        },
        error: function(xhr, status, error) {
            // Handle error
            console.error('Error:', error);
        }
    });
}

// Function to display recommendations in HTML
function displayRecommendations(recommendations) {
    // Clear previous recommendations
    $('#recommendations').empty();

    // Iterate over recommendations and append them to HTML
    for (var i = 0; i < recommendations.length; i++) {
        var recommendation = recommendations[i];
        $('#recommendations').append('<li>' + recommendation + '</li>');
    }    
}

// Function to handle form submission
$('#recommendForm').submit(function(event) {
    // Prevent default form submission
    event.preventDefault();

    // Get user ID from input field
    var userId = $('#userId').val();

    // Call function to make AJAX request
    getRecommendations(userId);
});

// Define categories
var categories = {
    // Define your categories here
};

// Function to extract user ID from the sentence using regex
var extractUserId = function(text) {
    var match = text.match(/\b\d+\b/);
    if (match) {
        return parseInt(match[0]);
    }
    return null;
};

// Function to handle chatbot interaction
var handleInteraction = function(userInput, conversation, categories, conversationState) {
    // Handle interaction logic here
};

// Client-side JavaScript for sending messages and displaying responses
function sendMessage() {
    var userInput = document.getElementById("userId").value; // Get user ID input
    var chatBody = document.getElementById("chat-body");

    // Create a message element for the user input
    var userMessageElement = document.createElement("div");
    userMessageElement.className = "message user-message";
    userMessageElement.textContent = "User ID: " + userInput;
    chatBody.appendChild(userMessageElement);

    // Send user message to backend using fetch API
    fetch("/recommend", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ user_id: userInput })
    })
    .then(response => response.json()) // Parse JSON response
    .then(data => {
        // Process recommendations from backend response
        var recommendationsElement = document.createElement("div");
        recommendationsElement.className = "message bot-message";
        recommendationsElement.textContent = "Recommendations: " + data;
        chatBody.appendChild(recommendationsElement);
        // Scroll to bottom of chat body
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
        console.error("Error:", error);
    });

    // Clear user input field
    document.getElementById("userId").value = "";
}

function handleBotResponse(botResponse) {
    var chatBody = document.getElementById("chat-body");
    var botResponseElement = document.createElement("div");
    botResponseElement.className = "message bot-message";
    botResponseElement.textContent = botResponse;
    chatBody.appendChild(botResponseElement);
    // Scroll to bottom of chat body
    chatBody.scrollTop = chatBody.scrollHeight;
}










// Define categories
var categories = {
    'Batteries': {
        'available': ['AAA Batteries (4-pack)', 'AA Batteries (4-pack)'],
        'unavailable': []
    },
    'Headphones': {
        'available': ['Wired Headphones', 'Apple Airpods Headphones', 'Bose SoundSport Headphones'],
        'unavailable': []
    },
    'Smart TV': {
        'available': ['27in FHD Monitor', '27in 4K Gaming Monitor', '34in Ultrawide Monitor', 'Flatscreen TV', '20in Monitor'],
        'unavailable': []
    },
    'Smart Phones': {
        'available': ['iPhone', 'Google Phone', 'Vareebadd Phone'],
        'unavailable': ['Samsung Galaxy', 'Xiaomi Phone']
    },
    'Laptops': {
        'available': ['Macbook Pro Laptop', 'ThinkPad Laptop'],
        'unavailable': []
    },
    'Cleaning Machines': {
        'available': ['LG Washing Machine', 'LG Dryer'],
        'unavailable': []
    }
};

// Function to extract user ID from the sentence using regex
function extractUserId(text) {
    var match = text.match(/\b\d+\b/);
    if (match) {
        return parseInt(match[0]);
    }
    return null;
}

// Function to handle chatbot interaction
function handleInteraction(userInput) {
    // Check if the user input contains any product names
    var mentionedProducts = [];
    for (var category in categories) {
        mentionedProducts.push(...categories[category].available.filter(product => userInput.toLowerCase().includes(product.toLowerCase())));
        mentionedProducts.push(...categories[category].unavailable.filter(product => userInput.toLowerCase().includes(product.toLowerCase())));
    }

    if (mentionedProducts.length > 0) {
        mentionedProducts.forEach(product => {
            for (var category in categories) {
                if (categories[category].available.includes(product)) {
                    conversation.push(`\nChatbot: Yes, ${product} is available.\n\n`);
                    break;
                } else if (categories[category].unavailable.includes(product)) {
                    var availableAlternatives = categories[category].available.join(', ');
                    conversation.push(`\nChatbot: No, ${product} is not available. Here are some alternatives from the ${category} category: ${availableAlternatives}.\n\n`);
                    break;
                }
            }
        });
        conversation.push("\nChatbot: Is there anything else I can assist you with? (Yes/No)\n\n");
        conversationState = "WAITING_FOR_USER_NEEDS";
        return;
    }

    // Check if user input includes "recommend"
    if (userInput.toLowerCase().includes("recommend")) {
        if (conversationState === "INITIAL" || conversationState === "WAITING_FOR_RECOMMENDATION_OR_PRODUCT") {
            conversation.push("\nChatbot: Sure! Please provide your user ID.\n\n");
            conversationState = "WAITING_FOR_USER_ID";
        } else {
            conversation.push("\nChatbot: Sorry, I can only process recommendation requests at this time. Can you please rephrase or wait for the current interaction to finish?\n\n");
        }
        return;
    }

    // Handle other conversation states and inputs
if (conversationState === "WAITING_FOR_USER_ID") {
    var userId = extractUserId(userInput);
    if (userId) {
        var model = loadModel(); // Load model outside to avoid loading it multiple times
        var productNames = getProductNames();
        // Assuming df is defined elsewhere in your code
        var allProductIds = df['Product ID'].unique(); // Adjust this according to your data structure
        var recommendations = getRecommendations(userId, model, allProductIds, df, productNames); // Implement getRecommendations function
        conversation.push("\nChatbot: Here are some personalized product recommendations for you:\n\n");
        recommendations.forEach(row => {
            conversation.push(`${row['Product Name']}                ${row['Predicted Interaction']}\n`);
        });
        conversation.push("\nChatbot: Is there anything else I can assist you with? (Yes/No)\n\n");
        conversationState = "WAITING_FOR_YES_OR_NO";
    } else {
        conversation.push("\nChatbot: Please enter a valid user ID.\n\n");
    }
} else if (conversationState === "WAITING_FOR_YES_OR_NO") {
    if (userInput.toLowerCase().includes("yes")) {
        conversation.push("\nChatbot: Do you want product recommendations or want to ask about products? (Recommendations/Products)\n\n");
        conversationState = "WAITING_FOR_RECOMMENDATION_OR_PRODUCT";
    } else if (userInput.toLowerCase().includes("no")) {
        conversation.push("\nChatbot: Thank you! If you need further assistance, feel free to ask.\n\n");
        conversationState = "INITIAL";
    } else {
        conversation.push("\nChatbot: Sorry, I did not quite get that. Can you please rephrase?\n\n");
    }
} else if (conversationState === "WAITING_FOR_USER_NEEDS") {
    if (userInput.toLowerCase().includes("yes")) {
        conversation.push("\nChatbot: Do you want product recommendations or want to ask about products? (Recommendations/Products)\n\n");
        conversationState = "WAITING_FOR_RECOMMENDATION_OR_PRODUCT";
    } else if (userInput.toLowerCase().includes("no")) {
        conversation.push("\nChatbot: Thank you! If you need further assistance, feel free to ask.\n\n");
        conversationState = "INITIAL";
    } else {
        conversation.push("\nChatbot: Sorry, I did not quite get that. Can you please rephrase?\n\n");
    }
} else if (conversationState === "WAITING_FOR_RECOMMENDATION_OR_PRODUCT") {
    if (userInput.toLowerCase().includes("recommendations")) {
        conversation.push("\nChatbot: Please provide your user ID.\n\n");
        conversationState = "WAITING_FOR_USER_ID";
    } else if (userInput.toLowerCase().includes("products")) {
        conversation.push("\nChatbot: Do you want to ask about specific categories or specific products? (Categories/Products)\n\n");
        conversationState = "WAITING_FOR_CATEGORY_OR_PRODUCT";
    } else {
        // Check if the user mentioned a specific product
        var productAvailable = false;
        for (var productName in productNames) {
            if (userInput.toLowerCase().includes(productNames[productName].toLowerCase())) {
                productAvailable = true;
                conversation.push(`\nChatbot: Yes, ${productNames[productName]} is available.\n\n`);
                break;
            }
        }
        if (!productAvailable) {
            // Check if the user mentioned a category of products
            var categoryMentioned = false;
            for (var category in categories) {
                if (userInput.toLowerCase().includes(category.toLowerCase())) {
                    categoryMentioned = true;
                    conversation.push(`\nChatbot: ${category} products available:\n`);
                    conversation.push(`${categories[category].available.join(', ')}\n\n`);
                    break;
                }
            }
            if (!categoryMentioned) {
                conversation.push("\nChatbot: Sorry, I couldn't find any relevant products. Please try again.\n\n");
            }
        }
        conversationState = "INITIAL";
    }
} else if (conversationState === "WAITING_FOR_CATEGORY_OR_PRODUCT") {
    if (userInput.toLowerCase().includes("categories")) {
        conversation.push("\nChatbot: What category of products are you interested in? (Enter category name)\n\n");
        conversationState = "WAITING_FOR_CATEGORY";
    } else if (userInput.toLowerCase().includes("products")) {
        conversation.push("\nChatbot: Please specify the product you want to ask about.\n\n");
        conversationState = "INITIAL";
    } else {
        conversation.push("\nChatbot: Sorry, I did not quite get that. Can you please rephrase?\n\n");
    }
} else if (conversationState === "WAITING_FOR_CATEGORY") {
    var selectedCategory = null;
    for (var category in categories) {
        if (userInput.toLowerCase().includes(category.toLowerCase())) {
            selectedCategory = category;
            break;
        }
    }
    if (selectedCategory) {
        conversation.push(`\nChatbot: Here are the products in the ${selectedCategory} category:\n\n`);
        conversation.push(`${categories[selectedCategory].available.join(', ')}\n`);
        conversation.push("\nChatbot: Is there anything else I can assist you with? (Yes/No)\n\n");
        conversationState = "WAITING_FOR_YES_OR_NO";
    } else {
        conversation.push("\nChatbot: Please select a valid category.\n\n");
    }
}

}

// Load model
function loadModel() {
    // Implement loading model logic here
}

// Load product names
function getProductNames() {
    var productNames = {
        1: 'iPhone',
        2: 'Lightning Charging Cable',
        3: 'Wired Headphones',
        4: '27in FHD Monitor',
        5: 'AAA Batteries (4-pack)',
        6: '27in 4K Gaming Monitor',
        7: 'USB-C Charging Cable',
        8: 'Bose SoundSport Headphones',
        9: 'Apple Airpods Headphones',
        10: 'Macbook Pro Laptop',
        11: 'Flatscreen TV',
        12: 'Vareebadd Phone',
        13: 'AA Batteries (4-pack)',
        14: 'Google Phone',
        15: '20in Monitor',
        16: '34in Ultrawide Monitor',
        17: 'ThinkPad Laptop',
        18: 'LG Dryer',
        19: 'LG Washing Machine',
        // Add more product IDs and names as needed
    };
    return productNames;
}

// Define global variables to manage conversation state
var conversation = [];
var conversationState = "INITIAL";














// Function to handle form submission
$('#recommendForm').submit(function(event) {
    // Prevent default form submission
    event.preventDefault();

    // Get user input from input field
    var userInput = $('#userInput').val();

    // Send user input to server for processing
    sendMessage(userInput);
});

// Function to send user input to the server and receive bot response
function sendMessage(userInput) {
    // Make AJAX POST request to /chat endpoint
    $.ajax({
        url: '/chat',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ user_input: userInput }),
        success: function(response) {
            // Handle successful response
            console.log('Bot Response:', response); // Log bot response to console
            displayBotResponse(response); // Display bot response on UI
        },
        error: function(xhr, status, error) {
            // Handle error
            console.error('Error:', error);
        }
    });
}

// Function to display bot response in HTML
function displayBotResponse(response) {
    // Append bot response to chat area
    $('#chat-area').append('<div class="message bot-message">' + response + '</div>');

    // Scroll to bottom of chat area
    var chatArea = document.getElementById('chat-area');
    chatArea.scrollTop = chatArea.scrollHeight;
}
