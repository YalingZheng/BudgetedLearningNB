package dk.blfw.core;



/**
 * An oracle interface is something that can answer a query.
 * It has three interface methods:<p>
 * 1.
 * answerQuery(Query) returns nonnull result for supported query, and null otherwise;
 * </p>
 * 2.
 * getSupportedQueries() return a list of supported query classes.
 * </p>
 * 3.
 * supportQuery(Query) returns true/false according to whether this oracle can handle the query
 * </p>
 * @author kdeng
 * 
 */
public interface IntfOracle {
	
	/**
	 * Answer a query q.
	 * If q is not supported by this particular oracle class, return null; otherwise the return is always nonnull. 
	 * Though not obligated, an oracle can use the constants defined in IntfAnswer 
	 * to give "nullanswer" as an answer.
	 * 
	 * @Pre  supportQuery(q)==true
	 * @Post result!=null
	 * 
	 * @param  q, something that can treated as a query.
	 * @return something meaningful to the user.
	 */
	IntfAnswer answerQuery(IntfQuery q);
	
	/**
	 * return a list of classes that are supported as queries 
	 * @return
	 */
	Class[] getSupportedQueries();
	
	/**
	 * return true if the dynamic type of q is supported. 
	 * 
	 * @Pre result==true implies q.class is one of the supported classes
	 * @param q
	 * @return
	 */
	boolean  supportQuery(Object q);
}
