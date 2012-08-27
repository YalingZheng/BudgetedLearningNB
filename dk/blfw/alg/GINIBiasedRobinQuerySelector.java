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

public class GINIBiasedRobinQuerySelector extends QuerySelector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final int seed=1234;
	private Random random= new Random(seed);
	
	private boolean inited=false;
	private int lastai=-1;
	
    private double lOld = -1;
    private double lNew = -1;
    private int whichAttribute = -1;
	
	public GINIBiasedRobinQuerySelector() {
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
			int ai= drawAttr(pool.numAttributes());
			int ii= drawInst(pool, c);

			if  (pool.instance(ii).isMissing(ai)) {
				lastai=ai;
				lOld = lNew;
				return new PhonyAttrQuery(ii,ai);
			}
		}			


		
	}

	private double calclNew(Instances pool, NaiveBayesAttrUpdatable c) {
		
		double PX = 1;
		double PYX = 1;
		double totalCounts = 0;
		double attrCounts = 1;
		double GINI = 0;
		
		//Compute P(x) probabilities
		double[][] GINIProbs = new double[pool.numAttributes()][];
		for(int i = 0; i < pool.numAttributes(); i++)
		{
			GINIProbs[i] = new double[pool.attribute(i).numValues()];
		}
		
		for(int atts = 0; atts < pool.numAttributes(); atts++)
		{
			for(int j = 0; j < pool.attribute(atts).numValues(); j++)
			{
				attrCounts = 1;
				totalCounts = pool.attribute(atts).numValues();
				Enumeration enu = pool.enumerateInstances();
				while(enu.hasMoreElements())
				{
					Instance inst = (Instance) enu.nextElement();
					if(!inst.isMissing(atts))
					{
						totalCounts += 1;
						if(inst.value(atts) == j)
						{
							attrCounts += 1;
						}
					}
				}
				GINIProbs[atts][j] = attrCounts / totalCounts;
			}
		}	
		
		//Generate instances for the GINI Index
		for(int count = 0; count < 400; count++)
		{
			Instance generated = pool.firstInstance();
			generated.setClassMissing();
			PX = 1;
			for(int atts = 0; atts < generated.numAttributes(); atts++)
			{
				double myRand = dk.blfw.Global.random.nextDouble();
				int myCount = 0;
				double partialPX = 0;
				if(generated.classIndex() != atts)
				{
					while(myRand > partialPX)
					{
						partialPX += GINIProbs[atts][myCount];
						myCount++;
					}
					generated.setValue(atts, myCount-1);
					PX = PX * partialPX;
				}
			}
			
			try
			{
				double PYXArray[] = c.distributionForInstance(generated);
				for(int y = 0; y < pool.numClasses(); y++)
				{
					PYX = PYXArray[y];
					GINI += PX*PYX*(1-PYX);
				}
			}
			catch (Exception e) 
			{
				e.printStackTrace();
			}
		}
		return GINI;
	}
	
	private int drawAttr(int numAttributes){
		if(lNew >= lOld)
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
}
