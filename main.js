function multiply_text() {
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
	textbox.value = out
}