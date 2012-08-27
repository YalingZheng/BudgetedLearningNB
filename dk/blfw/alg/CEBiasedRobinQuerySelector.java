package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.estimators.Estimator;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;
import dk.blfw.impl.NaiveBayesAttrUpdatable;

public class CEBiasedRobinQuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private Random random= new Random();
	
	private boolean inited=false;
	private int lastai=-1;
	
    private double lOld = -1;
    private double lNew = -1;
    private int whichAttribute = -1;
	
		
	public CEBiasedRobinQuerySelector() {
		super();
	}
	
	
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> context, Map optional) {		
		Instances pool= (Instances) context.get(QueryRequest.P);
		NaiveBayesAttrUpdatable c= (NaiveBayesAttrUpdatable) context.get(QueryRequest.C);
		
		if (!hasMissing(pool)) return IntfQuery.NONQUERY;
		
		
		if (!inited){
			inited=true;
			lOld = 1;
			lNew = 1;
			whichAttribute = 0;
		}	
		
		if (lastai>=0){
			//Calculate conditional entropy from last purchase
			lNew = calclNew(pool, c);
		}
		
		
		/**
		 * quit util find a missing attribute
		 */
		while(true){
			int ai= drawAttr(pool.numAttributes(),pool);
			int ii= drawInst(pool, c);

			if  (pool.instance(ii).isMissing(ai)) {
				lastai=ai;
				lOld = lNew;
				return new PhonyAttrQuery(ii,ai);
			}
		}			


		
	}

	private double calclNew(Instances pool, NaiveBayesAttrUpdatable c) {
		
		Estimator classes = c.getClassDistribution();
		Estimator[][] array = c.getMDistributions();
		double PY = -1;
		double PX = -1;
		double PXY = -1;
		double PYX = -1;
		double logPYX = 0;
		double totalCounts = pool.attribute(whichAttribute).numValues();
		double attrCounts = 1;
		double CE = 1;
		
		for(int x = 0; x < pool.attribute(whichAttribute).numValues(); x++)
		{
			Enumeration enu = pool.enumerateInstances();
			while(enu.hasMoreElements())
			{
				Instance inst = (Instance) enu.nextElement();
				
				if(!inst.isMissing(whichAttribute))
				{
					totalCounts += 1;
					if(inst.value(whichAttribute) == x)
					{
						attrCounts += 1;
					}
				}
			}
			
			PX = attrCounts / totalCounts;
			attrCounts = 1;
			totalCounts = pool.attribute(whichAttribute).numValues();
			
			double innerSum = 1;
			for(int y = 0; y < pool.classAttribute().numValues(); y++)
			{
				PY = classes.getProbability(y);
				PXY = array[whichAttribute][y].getProbability(x);
				PYX = PXY*PY/PX;
				logPYX = Math.log(PYX)/Math.log(2);
				innerSum = innerSum + (PYX*logPYX);
			}
			CE = CE + (PX * innerSum);
		}
		CE = -CE;
		return CE;
	}
	
	private int drawAttr(int numAttributes, Instances p){
        if ((lNew>= lOld) || !hasMissing(p,whichAttribute))
	//	if(lNew >= lOld)
		{
			whichAttribute = (whichAttribute + 1) % numAttributes;
		}
		
		return whichAttribute;
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
        
        @SuppressWarnings("unchecked")
	private boolean hasMissing(Instances pool, int attr){
		boolean hasmissing=false;
		for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
			Instance i=e.nextElement();
			if (i.isMissing(attr)) {hasmissing=true; break;}
		}
		return hasmissing;
	}
}
