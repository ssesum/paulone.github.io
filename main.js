function multiply_text() {
	var in_text = document.forms['myForm']['input_text'].value;
	var count = document.forms['myForm']['multiply_count'].value;
	var out = ""
	
	
	for (i=0; i<count; i++){
		out += in_text;
		out += "\n";
	}
	
	textbox = document.getElementById('output');
	textbox.value = out
}