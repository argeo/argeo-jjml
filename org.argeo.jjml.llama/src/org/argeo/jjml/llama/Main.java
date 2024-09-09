package org.argeo.jjml.llama;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {

	public static void main(String[] args) {
		String modelPathStr;
		//modelPathStr = "/srv/ai/models/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf";
		modelPathStr = "/srv/ai/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
		
		Path modelPath = Paths.get(modelPathStr);
		
		LlamaCppBackend backend = new LlamaCppBackend();
		backend.init();
		
		LlamaCppModel model = new LlamaCppModel();
		model.setLocalPath(modelPath);
		model.init();
		
		LlamaCppContext context = new LlamaCppContext();
		context.setModel(model);
		context.init();
		
		// clean up
		context.destroy();
		model.destroy();
		backend.destroy();
	}

}
