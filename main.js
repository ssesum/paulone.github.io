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


