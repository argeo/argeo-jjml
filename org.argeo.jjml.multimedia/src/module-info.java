module org.argeo.jjml.multimedia {
	exports org.argeo.jjml.mtmd;
	exports org.argeo.jjml.whisper;
	
	exports org.argeo.jjml.mtmd.awt;
	exports org.argeo.jjml.whisper.sound;

	requires transitive java.desktop;

	requires transitive org.argeo.jjml;
}