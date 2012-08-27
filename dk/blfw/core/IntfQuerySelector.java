package dk.blfw.core;

import java.util.EnumMap;
import java.util.Map;

public interface IntfQuerySelector {

	public abstract IntfQuery propose(EnumMap<QueryRequest, Object> request,
			Map optional);
	
	public String getName();

}