package org.argeo.jjml.llama;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {

	public static void main(String[] args) {
		String modelPathStr;
		modelPathStr = "/srv/ai/models/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf";
		
		Path modelPath = Paths.get(modelPathStr);
		
		LlamaCppBackend backend = new LlamaCppBackend();
		backend.init();
		
		LlamaCppModel model = new LlamaCppModel();
		model.setLocalPath(modelPath);
		model.init();
		
		model.destroy();
		backend.destroy();
	}

}
