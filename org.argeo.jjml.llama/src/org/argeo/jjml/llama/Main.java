package org.argeo.jjml.llama;

public class Main {

	public static void main(String[] args) {
		LlamaCppBackend backend = new LlamaCppBackend();
		backend.init();
		
		backend.destroy();
	}

}
