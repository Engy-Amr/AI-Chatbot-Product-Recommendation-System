"use strict";

function _typeof(o) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (o) { return typeof o; } : function (o) { return o && "function" == typeof Symbol && o.constructor === Symbol && o !== Symbol.prototype ? "symbol" : typeof o; }, _typeof(o); }
var _jquery = _interopRequireDefault(require("jquery"));
var _spacy = require("spacy");
var natural = _interopRequireWildcard(require("natural"));
function _getRequireWildcardCache(e) { if ("function" != typeof WeakMap) return null; var r = new WeakMap(), t = new WeakMap(); return (_getRequireWildcardCache = function _getRequireWildcardCache(e) { return e ? t : r; })(e); }
function _interopRequireWildcard(e, r) { if (!r && e && e.__esModule) return e; if (null === e || "object" != _typeof(e) && "function" != typeof e) return { "default": e }; var t = _getRequireWildcardCache(r); if (t && t.has(e)) return t.get(e); var n = { __proto__: null }, a = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var u in e) if ("default" !== u && {}.hasOwnProperty.call(e, u)) { var i = a ? Object.getOwnPropertyDescriptor(e, u) : null; i && (i.get || i.set) ? Object.defineProperty(n, u, i) : n[u] = e[u]; } return n["default"] = e, t && t.set(e, n), n; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
// Import jQuery using ES6 module syntax
// Import jQuery using ES6 module syntax
// Import specific function from spacy
// Import entire natural module

// TensorFlow.js library can be included via CDN link in HTML
// No equivalent for 'canvas', consider alternative browser APIs

// Function to make AJAX request to Flask app
function getRecommendations(userId) {
  // Make AJAX POST request to /recommend endpoint
  _jquery["default"].ajax({
    url: '/recommend',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({
      user_id: userId
    }),
    success: function success(response) {
      // Handle successful response
      console.log('Recommendations:', response); // Log recommendations to console
      displayRecommendations(response);
    },
    error: function error(xhr, status, _error) {
      // Handle error
      console.error('Error:', _error);
    }
  });
}

// Function to display recommendations in HTML
function displayRecommendations(recommendations) {
  // Clear previous recommendations
  (0, _jquery["default"])('#recommendations').empty();

  // Iterate over recommendations and append them to HTML
  for (var i = 0; i < recommendations.length; i++) {
    var recommendation = recommendations[i];
    (0, _jquery["default"])('#recommendations').append('<li>' + recommendation + '</li>');
  }
}

// Function to handle form submission
(0, _jquery["default"])('#recommendForm').submit(function (event) {
  // Prevent default form submission
  event.preventDefault();

  // Get user ID from input field
  var userId = (0, _jquery["default"])('#userId').val();

  // Call function to make AJAX request
  getRecommendations(userId);
});

// Define categories
var categories = {
  // Define your categories here
};

// Function to extract user ID from the sentence using regex
var extractUserId = function extractUserId(text) {
  var match = text.match(/\b\d+\b/);
  if (match) {
    return parseInt(match[0]);
  }
  return null;
};

// Function to handle chatbot interaction
var handleInteraction = function handleInteraction(userInput, conversation, categories, conversationState) {
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
    body: JSON.stringify({
      user_id: userInput
    })
  }).then(function (response) {
    return response.json();
  }) // Parse JSON response
  .then(function (data) {
    // Process recommendations from backend response
    var recommendationsElement = document.createElement("div");
    recommendationsElement.className = "message bot-message";
    recommendationsElement.textContent = "Recommendations: " + data;
    chatBody.appendChild(recommendationsElement);
    // Scroll to bottom of chat body
    chatBody.scrollTop = chatBody.scrollHeight;
  })["catch"](function (error) {
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
