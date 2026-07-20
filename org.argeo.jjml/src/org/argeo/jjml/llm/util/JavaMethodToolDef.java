package org.argeo.jjml.llm.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executor;

import org.argeo.jjml.llm.instruct.LlamaCppToolDefinition;

/** Converts a Java method to a basic tool definition, using reflection. */
public class JavaMethodToolDef implements LlamaCppToolDefinition {

	private final Method method;

	private final String description;

	private Object defaultContext;

//	private final Map.Entry<String, String>[] params;

	/** {@link LinkedHashMap} in order to keep the order. */
	private final Map<String, Param> parameters = new LinkedHashMap<>();

	/**
	 * @param params <name,description>, in the same order as the method arguments.
	 */
	@SafeVarargs
	public JavaMethodToolDef(Method method, String description, Map.Entry<String, String>... params) {
		this.method = method;
		this.description = description;
		if (params.length != method.getParameterCount())
			throw new IllegalArgumentException(
					"Method " + method + " has " + method.getParameterCount() + " parameters");
		// Parameter[] parameters = method.getParameters();
		Class<?>[] parameterTypes = method.getParameterTypes();
		for (int i = 0; i < params.length; i++) {
			ParamType paramType = ParamType.fromClass(parameterTypes[i]);
			SimpleParam param = new SimpleParam(paramType, params[i].getValue());
			parameters.put(params[i].getKey(), param);
		}
	}

	@Override
	public String getDescription() {
		return description;
	}

	@Override
	public Map<String, Param> getParameters() {
		return Collections.unmodifiableMap(parameters);
	}

	public boolean isStatic() {
		return Modifier.isStatic(method.getModifiers());
	}

	public String getDefaultToolName() {
		String className = method.getDeclaringClass().getName();
		String methodName = method.getName();

		className = className.replace('.', '_').toLowerCase();
		methodName = methodName.toLowerCase();

		String toolName = className + "__" + methodName;
		return toolName;
	}

	public void setDefaultContext(Object defaultContext) {
		this.defaultContext = defaultContext;
	}

	// TODO find a better method name?
	public CompletionStage<Map<String, Object>> execute(Object context, Map<String, Object> args, Executor executor) {
		Object instance = context != null ? context : defaultContext;
		if (instance == null && !isStatic())
			throw new NullPointerException(
					"An instance of " + method.getDeclaringClass() + " must be provided in order to execute " + method);
		if (args == null && parameters.size() != 0)
			throw new IllegalArgumentException("Expecting " + parameters.size() + " arguments");
		Object[] methodArgs = new Object[parameters.size()];
		if (args != null) {
			if (args.size() != parameters.size())
				throw new IllegalArgumentException("Expecting " + parameters.size() + " arguments, not " + args.size());
			Class<?>[] parameterTypes = method.getParameterTypes();
			int index = 0;
			for (String paramName : parameters.keySet()) {
				if (!args.containsKey(paramName))
					throw new IllegalArgumentException("Parameter " + paramName + " not found in arguments");
				Object value = args.get(paramName);
				Objects.requireNonNull(value);
				Class<?> clss = parameterTypes[index];
				if (String.class.isAssignableFrom(clss))
					methodArgs[index] = value.toString();
				else if (Integer.class.isAssignableFrom(clss))
					methodArgs[index] = Integer.parseInt(value.toString());
				else if (Float.class.isAssignableFrom(clss))
					methodArgs[index] = Float.parseFloat(value.toString());
				else if (Double.class.isAssignableFrom(clss))
					methodArgs[index] = Double.parseDouble(value.toString());
				else
					// TODO make it extensible
					throw new UnsupportedOperationException("Parma of type " + clss + " are not supported.");
				index++;
			}
		}

		CompletableFuture<Map<String, Object>> res = CompletableFuture.supplyAsync(() -> {
			try {
				Object result = method.invoke(instance, methodArgs);
				if (result instanceof Map) {
					Map<?, ?> map = (Map<?, ?>) result;
					Map<String, Object> resMap = new LinkedHashMap<>();
					for (Object key : map.keySet()) {
						resMap.put(key.toString(), map.get(key));
					}
					return resMap;
				} else {
					// TODO make it more canonical
					return Collections.singletonMap("result", result);
				}
			} catch (IllegalAccessException | InvocationTargetException e) {
				throw new RuntimeException("Cannot run tool " + method, e);
			}
		}, executor);
		return res;
	}
}
