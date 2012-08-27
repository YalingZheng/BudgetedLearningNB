package dk.blfw.core;

import java.io.Serializable;

/**
 * A marker interface for anything that can be used as a query 
 * @author kdeng
 *
 */
public interface IntfQuery extends Serializable {
	
	/**
	 * A "null query" is returned when the query selector doesn't know what else to ask;
	 * This can happen when say, everything is fully specified etc.
	 * A null query should be regarded as a token for failure state from the query selector.
	 */
	public static final IntfQuery NONQUERY= new IntfQuery(){
		private static final long serialVersionUID = 1L;

		public boolean isNullQuery() {
			// TODO Auto-generated method stub
			return true;
		}
		
	};

	/**
	 * An "any query" is issued the query selector suggests "anything is good";
	 * This can happen when say, we want to initialize the classifier.
	 * 
	 */
//	public static final IntfQuery ANYQUERY= new IntfQuery(){
//	private static final long serialVersionUID = 1L;};
	public boolean isNullQuery();
}
