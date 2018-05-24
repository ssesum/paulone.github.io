/*
This is a list of personal functions I've created that have helped me in my work.
*/
function log(message){
	/*Helper function to log to the console messages for debugging.*/
	console.log(message);
}


function print_multiplied_text() {
	/*This function will multiply the text you input into the input box.*/
	var in_text = document.forms['myForm']['input_text'].value;
	var count = document.forms['myForm']['multiply_count'].value;
	var index_choice = document.querySelector('input[name="choice"]:checked').value;
	var out = ""
	
	if(index_choice == "no"){
		for (i=0; i<count; i++){
			out += in_text;
			out += "\n";
		}
	}else{
		for (i=0; i<count; i++){
			out += in_text;
			out += " ";
			out += i;
			out += "\n";
		}
	}
	

	textbox = document.getElementById('output');
	textbox.value = out;
}


function remove_new_lines() {
	/*This function will remove all new lines from the given input box.*/
	var regex = new RegExp('\n', 'g');
	var index_choice = document.querySelector('input[name="choice"]:checked').value;
	textarea_input = document.getElementById("input").value;
	
	if(index_choice == "none"){
		textarea_output = textarea_input.replace(regex, "");		
	} else if (index_choice == "tab"){
		textarea_output = textarea_input.replace(regex, "\t");
	} else {
		textarea_output = textarea_input.replace(regex, ",");	
	}

	textarea_out = document.getElementById('output');
	textarea_out.value = textarea_output;
}


function convert_list_to_rows() {
	/*Input a comma separated list and convert that to new line separated rows.*/
	textarea_input = document.getElementById("input").value;
	var regex = new RegExp(',', 'g');
	textarea_output = textarea_input.replace(regex, '\n')
	textarea_out = document.getElementById('output');
	textarea_out.value = textarea_output;	
}


function put_quotations_around_text() {
	/*Given any amount of comma separated or plain rows, you can put quotations around them.*/
	textarea_input = document.getElementById("input").value;
	if(textarea_input.includes(",")){
		var regex = new RegExp(',\n', 'g');		
	} else {
		var regex = new RegExp('\n', 'g');
	}

	textarea_output = textarea_input.split(regex);
	var array_length = textarea_output.length;
	var new_textarea_output = []
	for(i=0; i < array_length; i++){
		curr_element = "\n\"";
		curr_element += textarea_output[i];
		curr_element += "\"";
		new_textarea_output.push(curr_element)
	}
	new_textarea_output[0] = new_textarea_output[0].replace("\n","")
	textarea_out = document.getElementById('output');
	textarea_out.value = new_textarea_output;
}


function python_def_extraction() {
	/*Insert your python code and this program will print out all the rows that contain "def".*/
	textarea_input = document.getElementById("input").value;
	var regex = new RegExp('\n', 'g');
	textarea_output = textarea_input.split(regex);
	var array_length = textarea_output.length;
	var new_textarea_output = [];
	for(i=0; i < array_length; i++){
		curr_element = ""
		if (textarea_output[i].includes("def")){
			curr_element += textarea_output[i];
			curr_element += "\n";
			new_textarea_output.push(curr_element);
		}
	}
	textarea_out = document.getElementById('output');
	textarea_out.value = new_textarea_output.join("")
}


function common_value_printer() {
	/*Checks two comma separated list for any common terms.Then prints the output to the text area.*/
	set1 = document.getElementById("set1").value;
	set2 = document.getElementById("set2").value;
	set_1 = set1.split(",");
	set_2 = set2.split(",");
	common_set = [];
	for(i=0; i<set_1.length; i++){
		for(j=0; j<set_2.length; j++){
			if(set_1[i] == set_2[j]){
				common_set.push(set_1[i]);
			}
		}
	}
	textarea_out = document.getElementById('output');
	textarea_out.value = common_set;
}