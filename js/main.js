/**
This is a list of personal functions I've created that have helped me in my work.
*/
function log(message) {
	/*Helper function to log to the console messages for debugging.*/
	console.log(message);
}

function printTextMultiplied() {
	/*This function will multiply the text you input into the input box.*/
	var inText = document.forms['myForm']['input_text'].value;
	var count = document.forms['myForm']['multiply_count'].value;
	var indexChoice = document.querySelector('input[name="choice"]:checked').value;
	var out = ""
	if (indexChoice == "no") {
		for (i = 0; i < count; i++) {
			out += inText;
			out += "\n";
		}
	} else {
		for (i = 0; i < count; i++) {
			out += inText;
			out += " ";
			out += i;
			out += "\n";
		}
	}
	textbox = document.getElementById('output');
	textbox.value = out;
}

function removeNewLines() {
	/*This function will remove all new lines from the given input box.*/
	var regex = new RegExp('\n', 'g');
	var indexChoice = document.querySelector('input[name="choice"]:checked').value;
	textareaInput = document.getElementById("input").value;
	if (indexChoice == "none") {
		textareaOutput = textareaInput.replace(regex, "");
	} else if (indexChoice == "tab") {
		textareaOutput = textareaInput.replace(regex, "\t");
	} else {
		textareaOutput = textareaInput.replace(regex, ",");
	}
	textareaOut = document.getElementById('output');
	textareaOut.value = textareaOutput;
}

function convertListsToRows() {
	/*Input a comma separated list and convert that to new line separated rows.*/
	textareaInput = document.getElementById("input").value;
	var regex = new RegExp(',', 'g');
	textareaOutput = textareaInput.replace(regex, '\n')
	textareaOut = document.getElementById('output');
	textareaOut.value = textareaOutput;
}

function putQuotationsAroundText() {
	/*Given any amount of comma separated or plain rows, you can put quotations around them.*/
	textareaInput = document.getElementById("input").value;
	if (textareaInput.includes(",")) {
		var regex = new RegExp(',\n', 'g');
	} else {
		var regex = new RegExp('\n', 'g');
	}
	textareaOutput = textareaInput.split(regex);
	var arrayLength = textareaOutput.length;
	var newTextAreaOutput = []
	for (i = 0; i < arrayLength; i++) {
		currElement = "\n\"";
		currElement += textareaOutput[i];
		currElement += "\"";
		newTextAreaOutput.push(currElement)
	}
	newTextAreaOutput[0] = newTextAreaOutput[0].replace("\n", "")
	textareaOut = document.getElementById('output');
	textareaOut.value = newTextAreaOutput;
}

function extractPythonDefinitions() {
	/*Insert your python code and this program will print out all the rows that contain "def".*/
	textareaInput = document.getElementById("input").value;
	var regex = new RegExp('\n', 'g');
	textareaOutput = textareaInput.split(regex);
	var arrayLength = textareaOutput.length;
	var newTextAreaOutput = [];
	for (i = 0; i < arrayLength; i++) {
		currElement = ""
		if (textareaOutput[i].includes("def")) {
			currElement += textareaOutput[i];
			currElement += "\n";
			newTextAreaOutput.push(currElement);
		}
	}
	textareaOut = document.getElementById('output');
	textareaOut.value = newTextAreaOutput.join("")
}

function printCommonValues() {
	/*Checks two comma separated list for any common terms.Then prints the output to the text area.*/
	set1 = document.getElementById("set1").value;
	set2 = document.getElementById("set2").value;
	setOneSplit = set1.split(",");
	setTwoSplit = set2.split(",");
	commonSet = [];
	for (i = 0; i < setOneSplit.length; i++) {
		for (j = 0; j < setTwoSplit.length; j++) {
			if (setOneSplit[i] == setTwoSplit[j]) {
				commonSet.push(setOneSplit[i]);
			}
		}
	}
	textareaOut = document.getElementById('output');
	textareaOut.value = commonSet;
}