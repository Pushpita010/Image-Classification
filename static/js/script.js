// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const classifyBtn = document.getElementById("classifyBtn");
const modelSelect = document.getElementById("modelSelect");
const loadingText = document.getElementById("loadingText");
const resultsSection = document.getElementById("resultsSection");
const errorSection = document.getElementById("errorSection");
const errorText = document.getElementById("errorText");

let selectedFile = null;

// ===== File Upload Handling =====
uploadArea.addEventListener("click", () => imageInput.click());

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileSelect(files[0]);
  }
});

imageInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  // Validate file type
  const validTypes = [
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/webp",
  ];

  if (!validTypes.includes(file.type)) {
    showError("Invalid file type. Please upload a valid image file.");
    return;
  }

  // Validate file size (max 16MB)
  if (file.size > 16 * 1024 * 1024) {
    showError("File size is too large. Maximum size is 16MB.");
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewContainer.style.display = "block";
    uploadArea.style.display = "none";
    classifyBtn.disabled = !modelSelect.value;
    hideError();
  };
  reader.readAsDataURL(file);
}

function removeImage() {
  selectedFile = null;
  imageInput.value = "";
  previewContainer.style.display = "none";
  uploadArea.style.display = "block";
  classifyBtn.disabled = true;
  resultsSection.style.display = "none";
  hideError();
}

// ===== Model Selection =====
function loadModels() {
  fetch("/models")
    .then((response) => response.json())
    .then((data) => {
      const models = data.models;

      // Clear existing options
      modelSelect.innerHTML = '<option value="">Select a model...</option>';

      // Add models to dropdown
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.name;
        modelSelect.appendChild(option);
      });

      // Add change event listener
      modelSelect.addEventListener("change", () => {
        classifyBtn.disabled = !selectedFile || !modelSelect.value;
      });
    })
    .catch((error) => {
      console.error("Error loading models:", error);
      modelSelect.innerHTML = '<option value="">Error loading models</option>';
    });
}

// ===== Image Classification =====
async function classifyImage() {
  if (!selectedFile) {
    showError("Please select an image first.");
    return;
  }

  if (!modelSelect.value) {
    showError("Please select a model.");
    return;
  }

  try {
    // Show loading state
    classifyBtn.disabled = true;
    loadingText.style.display = "flex";
    resultsSection.style.display = "none";
    hideError();

    // Create form data
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", modelSelect.value);

    // Send request
    const response = await fetch("/classify", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok) {
      displayResults(result);
    } else {
      showError(result.error || "Classification failed. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("An error occurred during classification. Please try again.");
  } finally {
    classifyBtn.disabled = false;
    loadingText.style.display = "none";
  }
}

function displayResults(result) {
  // Hide error and results
  hideError();
  resultsSection.style.display = "block";

  // Update result
  const resultClass = document.getElementById("resultClass");
  const resultText = document.getElementById("resultText");
  const usedModel = document.getElementById("usedModel");

  resultClass.textContent = result.classification === "Dog" ? "🐕" : "🐱";
  resultText.textContent = `This is a ${result.classification}`;

  // Format model name
  const modelNames = {
    svm: "Support Vector Machine (SVM)",
    random_forest: "Random Forest",
    logistic_regression: "Logistic Regression",
    knn: "K-Nearest Neighbors (KNN)",
  };
  usedModel.textContent = modelNames[result.model_used] || result.model_used;

  // Display probabilities if available
  const probabilitiesSection = document.getElementById("probabilitiesSection");
  if (result.probabilities) {
    probabilitiesSection.style.display = "block";

    const catProb = result.probabilities.cat;
    const dogProb = result.probabilities.dog;

    // Update cat probability bar
    const catProbBar = document.getElementById("catProb");
    catProbBar.style.width = catProb * 100 + "%";
    document.getElementById("catProbText").textContent =
      (catProb * 100).toFixed(1) + "%";

    // Update dog probability bar
    const dogProbBar = document.getElementById("dogProb");
    dogProbBar.style.width = dogProb * 100 + "%";
    document.getElementById("dogProbText").textContent =
      (dogProb * 100).toFixed(1) + "%";
  } else {
    probabilitiesSection.style.display = "none";
  }
}

function resetForm() {
  removeImage();
  modelSelect.value = "";
}

// ===== Error Handling =====
function showError(message) {
  errorText.textContent = message;
  errorSection.style.display = "block";
}

function hideError() {
  errorSection.style.display = "none";
}

// ===== Download Result =====
function downloadResult() {
  const resultClass = document.getElementById("resultClass").textContent;
  const resultText = document.getElementById("resultText").textContent;
  const usedModel = document.getElementById("usedModel").textContent;

  const timestamp = new Date().toLocaleString();

  let content = `CAT VS DOG CLASSIFICATION RESULT\n`;
  content += `================================\n\n`;
  content += `Result: ${resultText}\n`;
  content += `Prediction Emoji: ${resultClass}\n`;
  content += `Model Used: ${usedModel}\n`;
  content += `Classification Time: ${timestamp}\n`;

  // Add probabilities if available
  const probabilitiesSection = document.getElementById("probabilitiesSection");
  if (probabilitiesSection.style.display !== "none") {
    const catProb = document.getElementById("catProbText").textContent;
    const dogProb = document.getElementById("dogProbText").textContent;
    content += `\nConfidence Scores:\n`;
    content += `Cat: ${catProb}\n`;
    content += `Dog: ${dogProb}\n`;
  }

  // Create download link
  const blob = new Blob([content], { type: "text/plain" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `classification_result_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

// ===== Initialize =====
document.addEventListener("DOMContentLoaded", () => {
  loadModels();

  // Check backend health
  fetch("/health")
    .then((response) => response.json())
    .then((data) => {
      if (!data.models_loaded) {
        showError(
          "Warning: Not all models are loaded. Please ensure model files are in the models/ directory.",
        );
      }
    })
    .catch((error) => console.error("Health check error:", error));
});
