package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;
import dk.blfw.impl.NaiveBayesAttrUpdatable;

public class RoundRobinQuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final int seed=1234;
	private Random random= new Random(seed);
	
	//private int count = 0;
	
	private boolean inited=false;
    private int whichAttribute = -1;
	
	
	public RoundRobinQuerySelector() {
		super();
	}
	
	
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		NaiveBayesAttrUpdatable c= (NaiveBayesAttrUpdatable) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		
		if (!inited){
			inited=true;
			whichAttribute = 0;
		}	
		
		whichAttribute = (whichAttribute + 1) % pool.numAttributes();	
		if(whichAttribute == pool.classIndex())
		{
			whichAttribute = (whichAttribute + 1) % pool.numAttributes();
		}
		//System.out.println(whichAttribute);
		//System.out.println(pool.attribute(whichAttribute).name());
		//System.out.println(pool.classAttribute().index());
		//System.out.println(pool.classIndex());
		//if(whichAttribute == 0)
		//	count++;
		//System.out.println(count);
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ii= drawInst(pool, c);

			if  (pool.instance(ii).isMissing(whichAttribute)) {
				return new PhonyAttrQuery(ii,whichAttribute);
			}
		}			


		
	}
	
	private int drawInst(Instances p, NaiveBayesAttrUpdatable c){
		double rand = dk.blfw.Global.random.nextDouble();
		int myLabel = 0;
		int i = 0;
		double soFar = 0;
		boolean found = false;
		while(!found)
		{
			soFar += c.getClassDistribution().getProbability(i);
			if(rand <= soFar)
			{
				myLabel = i;
				found = true;
			}
			i++;
		}
			
		int j = -1;
		while(true)
		{
			j = dk.blfw.Global.random.nextInt(p.numInstances());
			if  (p.instance(j).classValue() == myLabel) {
				return j;
			}	
		}
	}
	
	@SuppressWarnings("unchecked")
	private boolean hasMissing(Instances pool){
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.hasMissingValue()) {hasmissing=true; break;}
		}
		return hasmissing;
	}
}
