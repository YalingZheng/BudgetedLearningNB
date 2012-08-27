package dk.blfw.core;

/**
 * query context constants:
 * provides essential information and request for a query selector.
 * 
 * provided:
 * C: classifier
 * P: entire P
 * LABELED: indexes of already labeled for this selector
 * 
 * request:
 * return should satisify:
 * MIN_RETURN:  min number of atom queries that should be returned; //not used
 * MAX_RETURN:  max number of atom queries that should be returned; //not used
 * query selector should return as many as possible while satisfying the above constraints; else return null-query.
 * 
 * CANDIDATE:  indexes of candidates for this round;  it may have been filtered
 * DO_FILTER: should this query selector do closest filter first; //not used
 * NUM_FILTER: maximum size of candiate_set after filtering; //not used
 * 
 * @author kdeng
 *
 */
public enum QueryRequest {

	C,P,LABELED,MIN_RETURN, MAX_RETURN, CANDIDATE, DO_FILTER, NUM_FILTER;
}
