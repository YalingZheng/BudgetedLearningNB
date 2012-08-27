package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.IntfQuerySelector;
import dk.blfw.core.QueryRequest;
//import dk.blfw.impl.NaiveBayesAttrUpdatable;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;
import static dk.blfw.alg.ROW_SELECTION_TYPE.*;

public class RowUncertaintyQuerySelector extends QuerySelector {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private IntfQuerySelector selector = new RandomQuerySelector();
	
	private boolean labelSensitive = false;

    private ROW_SELECTION_TYPE rowSelectionType = NONE;

	public String getDefaultQuerySelctorName() {
		return RandomQuerySelector.class.getName();
	}
	
	@Override
	public IntfQuery propose(EnumMap<QueryRequest, Object> request, Map optional) {
		// TODO Auto-generated method stub
		
		//forward the query to selector
		IntfQuery a = selector.propose(request, optional);
		
		if (a.isNullQuery())
			return null;
		
		//fill in chris's code. 
		RowSelector MyRowSelector = new RowSelector();
		MyRowSelector.setSeed(dk.blfw.Global.random.nextInt());
		PhonyAttrQuery aa= (PhonyAttrQuery) a;		
		Instances pool= (Instances) request.get(QueryRequest.P);
		int desiredAttr = aa.getAttrIndex();
		//int desiredClass = pool.instance(aa.getInstIndex()).classIndex();
		Double classValue = pool.instance(aa.getInstIndex()).classValue();
		int desiredClass = classValue.intValue();
		//MyRowSelector.setSelectorType(RowSelector.RANDOM);
		Classifier c = (Classifier) request.get(QueryRequest.C);
		//System.out.println("desiredAttr="+desiredAttr+" desiredclass="+desiredClass);
		//System.out.println(pool.instance(aa.getInstIndex()).classValue());
		//System.out.println();
	    int row = -1;
	    if(labelSensitive)
	    {
	        switch(rowSelectionType) {
	        
	          case NONE:
	        	  return a;
	          case FIRST:
	        	  row = MyRowSelector.SelectRow_First(pool, desiredAttr, desiredClass);
	        	  break;
	          case RANDOM: 
	        	  //System.out.println("Random");
	        	  row = MyRowSelector.SelectRow_Random(pool, desiredAttr, desiredClass); 
	        	  break;
	          case KLDIVERGENCE: 
	        	  //System.out.println("KLDivergence");
	        	  row = MyRowSelector.SelectRow_KLDivergence(pool, c, desiredAttr, desiredClass); 
	        	  break;
	          case KLDIVMISCLASSIFIED:
	        	  row = MyRowSelector.SelectRow_KLDivergenceMisclassified(pool, c, desiredAttr, desiredClass);
	        	  break;
	          case L2NORM: 
	        	  //System.out.println("L2Norm");
	        	  row = MyRowSelector.SelectRow_L2Norm(pool, c, desiredAttr, desiredClass); 
	        	  break;
	          case ERRORMARGIN:
	        	  row= MyRowSelector.SelectRow_ErrorMargin(pool, c, desiredAttr,desiredClass);
	        	  break;
	          default: 
	        	  return a;
	      }
	    }
	    else //labelInsensitive
	    {
	        switch(rowSelectionType) {
	        
	          case NONE:
	        	  return a;
	          case FIRST:
	        	  row = MyRowSelector.SelectRow_First(pool, desiredAttr);
	        	  break;
	          case RANDOM: 
	        	  //System.out.println("Random");
	        	  row = MyRowSelector.SelectRow_Random(pool, desiredAttr); 
	        	  break;
	          case KLDIVERGENCE: 
	        	  //System.out.println("KLDivergence");
	        	  row = MyRowSelector.SelectRow_KLDivergence(pool, c, desiredAttr); 
	        	  break;
	          case KLDIVMISCLASSIFIED:
	        	  row = MyRowSelector.SelectRow_KLDivergenceMisclassified(pool, c, desiredAttr, desiredClass);
	        	  break;
	          case L2NORM: 
	        	  //System.out.println("L2Norm");
	        	  row = MyRowSelector.SelectRow_L2Norm(pool, c, desiredAttr); 
	        	  break;
	          case ERRORMARGIN:
	        	  row= MyRowSelector.SelectRow_ErrorMargin(pool, c, desiredAttr);
	          default: 
	        	  return a;
	      }
	    	
	    }
        //System.out.println("row="+row);
	    if(row == -1) //RowSelector was unable to find a valid instance to buy, return the original
	    {
	    	System.out.println("ERROR: RowSelector unable to find valid row.  (attribute,label) = (" + desiredAttr +","+desiredClass+")");
	    	return a;
	    	//return IntfQuery.NONQUERY;
	    }
	    else
	    	{
	    	//System.out.println("row="+row+" attr="+desiredAttr);
	        return new PhonyAttrQuery(row,desiredAttr);}
	}

	@Override
	public String[] getOptions() {
	    String [] options;
	    options = new String[1];
	    options[0]="-bq "+ getSelector().getClass().getName();
	    return options;

	}

	@SuppressWarnings("unchecked")
	@Override
	public Enumeration listOptions() {
		// TODO Auto-generated method stub
		

	    Vector newVector = new Vector(1);

	    newVector.addElement(new Option(
		      "\tbase selector name\n"
		      + "\t that defines the base selector used by rowuncertainty",
		      "bq", 1, "-bq <value>"));
	    
	    newVector.addElement(new Option(
			      "\trow selection type\n"
			      + "\t that defines the criteria for choosing a row",
			      "rs", 1, "-rs [(None), First, Random, KLDiv, KLDivMisclass, L2Norm]"));

	    newVector.addElement(new Option(
			      "\tlabel sensitive type\n"
			      + "\t specifies if row choices should be sensitive to the instance label",
			      "labelSensitive", 1, "-labelSensitive [true, (false)]"));

	    Enumeration enu = super.listOptions();
	    while (enu.hasMoreElements()) {
	      newVector.addElement(enu.nextElement());
	    }
	    return newVector.elements();

	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		super.setOptions(options);
		if (Utils.getOptionPos("bq", options)>=0){
			String sel_name= (Utils.getOption("bq", options));
			String row_sel_type = (Utils.getOption("rs", options));
			String label_sensitive = (Utils.getOption("labelSensitive", options));
			if (sel_name.length() > 0) { 
			      
			      setSelector(QuerySelector.forName(sel_name,
							       options));
			    } else {
			    	setSelector(new RandomQuerySelector());
			    }
			
			if(row_sel_type.length() > 0) 
			{
				// set your row selection type
				if(row_sel_type.equalsIgnoreCase("None"))
					rowSelectionType = NONE;
				else if(row_sel_type.equalsIgnoreCase("First"))
					rowSelectionType = FIRST;
				else if(row_sel_type.equalsIgnoreCase("Random"))
					rowSelectionType = RANDOM;				
				else if(row_sel_type.equalsIgnoreCase("KLDiv"))
					rowSelectionType = KLDIVERGENCE;
				else if(row_sel_type.equalsIgnoreCase("KLDivMisclass"))
					rowSelectionType = KLDIVMISCLASSIFIED;
				else if(row_sel_type.equalsIgnoreCase("L2Norm"))
					rowSelectionType = L2NORM;
				else if (row_sel_type.equalsIgnoreCase("ErrorMargin"))
					rowSelectionType= ERRORMARGIN;
				else
					rowSelectionType = NONE;
			}
			if(label_sensitive.length() > 0) {
				if(label_sensitive.equalsIgnoreCase("true"))
					labelSensitive = true;
				else if(label_sensitive.equalsIgnoreCase("false"))
					labelSensitive = false;
				else
					labelSensitive = false;					
			}
		}

	}




	public IntfQuerySelector getSelector() {
		return selector;
	}




	public void setSelector(IntfQuerySelector selector) {
		this.selector = selector;
	}


	
	
	

}
