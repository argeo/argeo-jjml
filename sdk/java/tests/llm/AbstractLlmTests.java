//!/usr/bin/env -S java -ea -cp /usr/share/java/org.argeo.jjml.jar
package tests.llm;

import org.argeo.jjml.llm.LlamaCppModel;

import tests.AbstractJjmlTests;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
abstract class AbstractLlmTests extends AbstractJjmlTests {
	private int parallelism = Runtime.getRuntime().availableProcessors();

	private final LlamaCppModel model;

	AbstractLlmTests(LlamaCppModel model) {
		this.model = model;
	}

	/*
	 * ACCESSORS
	 */
	LlamaCppModel getModel() {
		return model;
	}

	protected int getParallelism() {
		return parallelism;
	}

}
