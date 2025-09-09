package org.argeo.jjml.mtmd;

import org.argeo.jjml.llm.LlamaCppNative;

public class MtmdNative {
	private final static String JJML_MTMD_LIBRARY_NAME = "Java_" + MtmdNative.class.getPackageName().replace('.', '_');

	private static boolean librariesLoaded = false;
	/*
	 * STATIC UTILITIES
	 */

	public static void ensureLibrariesLoaded() {
		if (librariesLoaded)
			return;
		LlamaCppNative.ensureLibrariesLoaded();
		loadLibraries();

		System.loadLibrary(JJML_MTMD_LIBRARY_NAME);
	}

	static void loadLibraries() {
		checkLibrariesNotLoaded();
	}

	/** Fails if libraries already loaded. */
	private static void checkLibrariesNotLoaded() {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
	}

	/** singleton */
	private MtmdNative() {
	}
}
