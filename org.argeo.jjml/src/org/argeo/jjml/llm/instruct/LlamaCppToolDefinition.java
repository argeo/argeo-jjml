package org.argeo.jjml.llm.instruct;

import java.math.BigInteger;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;

public interface LlamaCppToolDefinition {
	String getDescription();

	Map<String, Param> getParameters();

	CompletionStage<Map<String, Object>> execute(Object context, Map<String, Object> args, Executor executor);

	default CompletionStage<Map<String, Object>> execute(Object context, Map<String, Object> args) {
		return execute(context, args, ForkJoinPool.commonPool());
	}

	default CompletionStage<Map<String, Object>> execute(Map<String, Object> args) {
		return execute(null, args, ForkJoinPool.commonPool());
	}

	default String apply(Map<String, Object> args) {
		Object value = execute(args).thenApply((res) -> {
			if (res.size() == 0)
				return null;
			if (res.size() > 1)
				throw new IllegalStateException("Expected only one unstructured result");
			return res.values().iterator().next();
		});
		if (value == null)
			return null;
		return value.toString();
	}
	
	static interface Param {
		ParamType getType();

		String getDescription();

		// TODO support enum, arrays, object, etc.
	}

	static enum ParamType implements Supplier<String> {
		STRING, //
		NUMBER, //
		INTEGER, //
		BOOLEAN, //
		ARRAY, //
		OBJECT, //
		;

		private final String type;

		private ParamType() {
			this.type = name().toLowerCase();
		}

		@Override
		public String get() {
			return type;
		}

		public static ParamType fromClass(Class<?> clss) {
			if (Boolean.class.isAssignableFrom(clss))
				return BOOLEAN;
			if (clss.isArray())
				return ARRAY;
			if (Number.class.isAssignableFrom(clss)) {
				if (Integer.class.isAssignableFrom(clss) //
						|| Short.class.isAssignableFrom(clss) //
						|| BigInteger.class.isAssignableFrom(clss))
					return INTEGER;
				return NUMBER;
			}
			// TODO deal with OBJECT
			return STRING;
		}

//		public Object toValue(String str) {
//			switch (this) {
//			case BOOLEAN:
//				return Boolean.parseBoolean(str);
//			case INTEGER:
//				return Boolean.parseBoolean(str);
//
//			default:
//				throw new UnsupportedOperationException("Unsupported type");
//			}
//		}
	}

	static class SimpleParam implements Param {
		private final ParamType type;
		private final String description;

		public SimpleParam(ParamType type, String description) {
			this.type = type;
			this.description = description;
		}

		public ParamType getType() {
			return type;
		}

		public String getDescription() {
			return description;
		}

	}
}
