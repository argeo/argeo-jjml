module org.argeo.jjml.llama {
	exports org.argeo.jjml.ggml.params;
	exports org.argeo.jjml.llama;
	exports org.argeo.jjml.llama.params;
	exports org.argeo.jjml.llama.util;

	// optional dependencies
	requires static java.management;
	requires static jdk.management;
	requires static jdk.httpserver;
}