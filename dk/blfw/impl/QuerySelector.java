package dk.blfw.impl;

import java.io.Serializable;
import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.IntfQuerySelector;
import dk.blfw.core.QueryRequest;


import weka.core.OptionHandler;
import weka.core.SerializedObject;
import weka.core.Utils;



/**
 * an abstract query selector that has default constructor 
 * @author kdeng
 *
 */
public abstract class QuerySelector  implements OptionHandler, IntfQuerySelector, Serializable {
	


	public String[] getOptions() {
		// TODO Auto-generated method stub
		return null;
	}



	public Enumeration listOptions() {
		// TODO Auto-generated method stub
		return new Vector().elements();
	}



	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		
	}



	/* (non-Javadoc)
	 * @see dk.blfw.core.IntfQuerySelector#propose(java.util.EnumMap, java.util.Map)
	 */
	public abstract IntfQuery propose(EnumMap<QueryRequest, Object> request, Map optional) ;



	public QuerySelector(){
		
	}
	
	public  static IntfQuerySelector forName(String classifierName, String[] options ){
		try {
			return (IntfQuerySelector)Utils.forName(IntfQuerySelector.class,
			     classifierName,
			     options);

		}catch (Exception e){
			return null;
		}
	
	}

	public static IntfQuerySelector makeCopy(IntfQuerySelector model) throws Exception {

	    return (IntfQuerySelector)new SerializedObject(model).getObject();
	}

	public String getName() {
		// TODO Auto-generated method stub
		return getClass().getSimpleName();
	}
	
//	//todo: hacked.  
//	@SuppressWarnings("unchecked")
//	public static Object forName(Class classType,
//			String className,
//			String [] options) throws Exception {
//
//		Class c = null;
//		try {
//			c = Class.forName(className);
//		} catch (Exception ex) {
//			throw new Exception("Can't find class called: " + className);
//		}
//		if (!classType.isAssignableFrom(c)) {
//			throw new Exception(classType.getName() + " is not assignable from "
//					+ className);
//		}
//		Object o = c.newInstance();
//		if ((o instanceof OptionHandler)
//				&& (options != null)) {
//			((OptionHandler)o).setOptions(options);
//			//Utils.checkForRemainingOptions(options);
//		}
//		return o;
//	}
}
