/***********************************************************************
 *
 *  UNIVERSVM IMPLEMENTATION
 *  Copyright (C) 2005 Fabian Sinz  Max Planck Institute for Biological Cybernetics
 *
 *  Includes parts of LUSH Lisp Universal Shell:
 *  Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *
 *  Includes parts of libsvm.
 *
 *
 *  This program is free software except for military or military related use; 
 *  If you don't come under the above exception you can redistribute it 
 *  and/or modify it under the terms of the GNU General Public License as 
 *  published by the Free Software Foundation; either version 2 of the 
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 ***********************************************************************/
 


#include <stdio.h>
#include <vector>
#include <math.h>

#include "svqp2/svqp2.h"
#include "svqp2/vector.h"

#ifdef MEX
#include "mex.h"
#endif

// various kernels 
#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3 
#define CUSTOM 4

// different univserum training algorithms
#define MULTIUNI 1
#define RAMPUNI 2

char *kernel_type_table[] = {"linear","polynomial","rbf","sigmoid","custom"};
char *optimizer_table[] = {"svqp2","cccp"};

#define SVQP 0
#define CCCP 1
#define ONLINE 1
#define ONLINE_WITH_FINISHING 2 
#define VERYBIG pow(10.0,100)
#define CCCP_TOL 0.0
#define TRAIN 0
#define TEST  1
#define UNIVERSUM 2
#define UNLABELED 3
#define SV 4

#define TRAINSAMP 0
#define UNIVESAMP 1
#define UNLABSAMP 2
#define EXTRASAMP 4


#define MAX_LOOPS 20 // maximum number of loops for tsvm training, should converge before this value, anyway..

#include <fstream>
#include <ctime>
#include <iostream>


/*********************** Split file storage class ******************/

class ID // class to hold split file indices and labels
{
public:
	int x;
	int y;
    ID() : x(0), y(0) {}
    ID(int x1,int y1) : x(x1), y(y1) {}
};
// IDs will be sorted by index, not by label.
bool operator<(const ID& x, const ID& y)
{
    return x.x < y.x;
}

/*********************** Little Stopwatch class ******************/
using namespace std;
class stopwatch
{
public:
 stopwatch() : start(std::clock()){} //start counting time
 ~stopwatch();
 double get_time(){
     clock_t total = clock()-start;;
     return double(total)/CLOCKS_PER_SEC;
 };

 void reset(){
     start=  clock();
 };
private:
 std::clock_t start;
};
stopwatch::~stopwatch()
{
// clock_t total = clock()-start; //get elapsed time
 //cout<<"total of ticks for this activity: "<<total<<endl;
 //cout<<"Time(secs): "<<double(total)/CLOCKS_PER_SEC<<endl;
}

/*******************************************************************/



/**********************  Variable Declarations *****************************/
double gap=0.05;                  // gap for universum
double s_ramp = -1;               // parameter for ramp loss
double s_trans = 0;               // parameter for transductive SVM
bool s_spec = 0;                  // did the user specify s?
int verbosity=1;                  // verbosity level, 0=off


/* Variable to store different parameters of data*/
double un_weight  = NAN;           // linear coefficient for the special kernel column
int mall;           		   // train+test size
int m=0;                           // training set size
int m_map[5];                      // map to the training set sizes
int max_index;                     // maximal index of the training/testing vectors
double maxa=0;						// largest value of alpha

/* Regularizer variables*/
double C=1;                     // C, penalty on errors
double C2=1;                    // C, penalty on universum
double C3=-1.0;                    // C, penalty on unlabeled
int ker_ridge = 0;

/* Variables for data*/
vector <lasvm_sparsevector_t*> X; // feature vectors
vector <int>  Y;                  // labels
vector < vector <int> >  multi_Y; // labels for multiclass
vector <double> kparam;           // kernel parameters
vector <double> x_square;         // norms of input vectors, used for RBF
vector <int> data_map[5];         // maps to the indices of train/test/... data
vector <double> global_ext_K;     // additional kernel column for transduction
vector <int> labels;              // all different labels, that occured in the datafiles

/* Variables for the model*/
vector <double> alpha;                   // alpha_i, SV weights
vector< vector<double> > multi_alpha;    // alpha_ij for multiclass
vector<double> multi_b0;                 // thresholds for multiclass
double b0;                               // threshold
int use_b0=1;                            // use threshold via constraint \sum a_i y_i =0
int kernel_type=LINEAR;                  // specifies the type of the kernel
vector <double> beta;                    // vector with betas for transduction
vector <double> Yest;                    // temporary vector to store the prediction
vector < vector<double> > D;             // 2D vector to store the function values of the test points
double C_star = VERYBIG;                 // C, penalty for the alpha of the special kernel column

/* Algorithm parameters*/
double degree=3,kgamma=2,coef0=0;        // kernel params
int cl=2;                                // number of classes
int optimizer=SVQP;                      // strategy of optimization
int folds=-1;                            // if folds=-1 --> no cross validation, else do folds fold CV
int use_ext_K = 0;                       // use extra kernel column?
int do_multi_class = 0;                  // do multiclass?
int prob_size = 0;                       // size of the problem (different to dataset size!)
int fill_level = 0;                      // temporary variable for setting up the problem

/* Programm behaviour*/
char report_file_name[1024];             // filename for the training report
char split_file_name[1024]="\0";         // filename for the splits
int cache_size=256;                      // 256Mb cache size as default
double epsgr=1e-3;                       // tolerance on gradients
long long kcalcs=0;                      // number of kernel evaluations
int binary_files=0;
int use_universum = 0;                   // specifies the training algorithm used for universum
vector <ID> splits; 
bool justtest = 0;                       // if justtest, then universvm tests on the specified model

/* Other */
int seed=0;
vector<double> training_time;
///****************************** Functions *************************************/


void exit_with_help()
{
	fprintf(stdout,
	"\nUSAGE: usvm [options] training_set_file [model_file]\n\n"
	"\n"
	"Trains a SVM, TSVM or USVM with specified parameters. If a model_file is\n"
	"specified after the training_set_file, then the learnt model will be stored\n"
	"there. If the model_file is supplied by the switch -F, then UniverSVM will\n"
	"test the specified model on the data that is supplied as training data and \n"
	"the test data supplied by -T. So\n"
	"   universvm [options] -T testfile trainfile\n"
	"has the same effect as doing\n"
	"   universvm [options] trainfile model \n"
	"   universvm [options] -F model testfile\n"
	"\n"
	"OPTIONS:\n"
	"FILE/DATA OPTIONS:\n"
	"-T test_set_file: test model on test set\n"
	"-U universum_file : use universum (it's also possible to include universum points \n"
	"                    with label -2 in the training file)\n"
	"-F model_file : Test the model stored in model_file on training AND test data (specified by -T)\n"
	"-u unlabeled_data_file : use unlabeled data (transductive SVM).\n"
	"	         (it's also possible to include unlabeled points\n"
	"                    with label -3 in the training file)\n"
	"-B file format : files are stored in the following format:\n"
	"	  0 -- libsvm ascii format (default)\n"
	"	  1 -- binary format\n"
	"	  2 -- split file format\n"
	"-f file : output report file to given destination\n"
	"-D file : output function values on test set(s) to given destination\n"
	"\nOPTIMIZATION OPTIONS:\n"
	"-V universum variant:\n" 
	"	  0 -- Standard universum training (default)\n"
	"	  1 -- Train SVM with universum by making it a 3-class multiclass\n"
	"	       problem and adding the decision rules for {+1,U} vs. -1 and\n"
	"	       {-1,U} vs. +1 (0=off default)\n"
	"	       This switch works only for binary at the moment.\n"
	"	  2 -- Train universum with ramp loss. This option requires \"-o 1\".\n"
	"-o optimizer: set different optimizers\n"
	"	  0 -- quadratic programm\n"
	"	  1 -- convex concave procedure (if you choose a transductive SVM,\n"
	"	       this option will be chosen automatically)\n"
	"-G gap : set gap parameter for universum (default 0.05) \n"
	"-I use_ridge : Add the ridge 1/C to the kernel matrix.\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC (default 1)\n"
	"-C cost : set the parameter C for universum points\n"
	"-a cost : set the parameter C for balancing constraint\n"
	"-z cost : set the parameter C for unlabeled points\n"
	"-m cachesize : set cache memory size in MB (default 256)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-s s : s parameter for ramp loss (default: -1 )\n"
	"-S s : s parameter for transductive SVM loss (default: 0)\n"
	"\nMODEL OPTIONS:\n"
	"-t kernel_type : set type of kernel function (default 0)\n"
	"	  0 -- linear: u'*v\n"
	"	  1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	  2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	  3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	  4 -- custom: k(x_i,x_j) = X(i,j)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/k)\n"
	"-b bias: use constraint sum alpha_i y_i =0 (default 1=on)\n"
        "-w weight: the rhs of the balancing constraint (default = sum(y_i)) \n"
	"-v n : do cross validation with n folds\n"
	"-M k : perform a multiclass training on k classes labeled with k different\n"
		"\tintegers >= 0 (default: 0)\n"
	);
	exit(1);
}


// ******************* Parse Functions **************************************
void parse_command_line(int argc, char **argv, char *input_file_name, char *universum_file_name, char *testset_file_name, char *unlabeled_file_name,  char *model_file_name, char *fval_file_name)
{
    int i; 

    // parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'o':
				optimizer = atoi(argv[i]);
				break;
			case 't':
				kernel_type = atoi(argv[i]);
				break;
			case 's':
				s_ramp = atof(argv[i]);
       				break;
			case 'S':
				s_trans = atof(argv[i]);
       				break;
		        case 'T':
			    strcpy(testset_file_name, argv[i]);
			    break;
		        case 'D':
			    strcpy(fval_file_name, argv[i]);
			    break;
		        case 'f':
			    strcpy(report_file_name, argv[i]);
			    break;
		        case 'P':
			    strcpy(split_file_name, argv[i]);
			    break;
		        case 'u':
			    strcpy(unlabeled_file_name, argv[i]);
			    break;
		        case 'U':
			    strcpy(universum_file_name, argv[i]);
			    break;
		        case 'V':
			    use_universum = atoi(argv[i]);
			    break;
		        case 'B':
			    binary_files= atoi(argv[i]);
			    break;
			case 'd':
				degree = atof(argv[i]);
				break;

			case 'g':
				kgamma = atof(argv[i]);
				break;
			case 'G':
				gap = atof(argv[i]);
				break;
			case 'I':
				ker_ridge = atoi(argv[i]);
				break;
			case 'r':
				coef0 = atof(argv[i]);
				break;
			case 'm':
				cache_size = (int) atof(argv[i]);
				break;
			case 'M':
				do_multi_class = (int) atof(argv[i]);
				break;
			case 'c':
				C = atof(argv[i]);
				break;
			case 'a':
				C_star = atof(argv[i]);
				break;
			case 'C':
				C2 = atof(argv[i]);
				break;
			case 'z':
				C3 = atof(argv[i]);
				break;
			case 'v':
				folds = atoi(argv[i]);
				break;
			case 'b':
				use_b0=atoi(argv[i]);
				break;
		        case 'w':
			        un_weight = atof(argv[i]);
				break;
			case 'e':
				epsgr = atof(argv[i]);
				break;
			case 'R':
				seed=atoi(argv[i]);
				break;
			case 'F':
			        strcpy(model_file_name,argv[i]);
				justtest = 1;
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}


	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	
	if(i<argc-1){
	  strcpy(model_file_name,argv[i+1]);
	}
	
	/*{ TRASH
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
		} TRASH END */

}




int split_file_load(char *f)
{
    int binary_file=0,labs=0,inds=0;
    FILE *fp;
    fp=fopen(f,"r"); 
    if(fp==NULL) {printf("[couldn't load split file: %s]\n",f); exit(1);}
    char dummy[100],dummy2[100];
    int i,j=0; 
    for(i=0;i<(int)strlen(f);i++) if(f[i]=='/') j=i+1;
    fscanf(fp,"%s %s",dummy,dummy2);
    strcpy(&(f[j]),dummy2);
    fscanf(fp,"%s %d",dummy,&binary_file);
    fscanf(fp,"%s %d",dummy,&inds);
    fscanf(fp,"%s %d",dummy,&labs);
    printf("[split file: load:%s binary:%d new_indices:%d new_labels:%d]\n",f,binary_file,inds,labs);
    //printf("[split file:%s binary=%d]\n",dummy2,binary_file);
    if(!inds) return binary_file;
    while(1)
    {
        int i,j;
        int c=fscanf(fp,"%d",&i);
        if(labs) c=fscanf(fp,"%d",&j);
        if(c==-1) break;
        if (labs) 
			splits.push_back(ID(i-1,j)); 
		else 
			splits.push_back(ID(i-1,0));
    }

    sort(splits.begin(),splits.end());
	
	return binary_file;
}


int libsvm_load_data(char *filename)
// loads the same format as LIBSVM
{
    int index; double value;
    int elements, i;
    FILE *fp = fopen(filename,"r");
    lasvm_sparsevector_t* v;

    if(fp == NULL)
    {
        fprintf(stderr,"Can't open input file \"%s\"\n",filename);
        exit(1);
    }
    else
        printf("loading \"%s\"..  \n",filename);
    int splitpos=0;

    int msz = 0; 
    elements = 0;
    while(1)
    {
        int c = fgetc(fp);
        switch(c)
        {
        case '\n':
            if(splits.size()>0) 
            {
                if(splitpos<(int)splits.size() && splits[splitpos].x==msz)
                {
                    v=lasvm_sparsevector_create();
                    X.push_back(v);	splitpos++;
                }
            }
            else
            {
                v=lasvm_sparsevector_create();
                X.push_back(v); 
            }
            ++msz;
            //printf("%d\n",m);
            elements=0;
            break;
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
 out:
    rewind(fp);


    max_index = 0;splitpos=0;
    for(i=0;i<msz;i++)
    {

        int write=0;
        if(splits.size()>0)
        {
            if(splitpos<(int)splits.size() && splits[splitpos].x==i)
            {
                write=2;splitpos++;
            }
        }
        else
            write=1; 

        int label;
        fscanf(fp,"%d",&label);
        //	printf("%d %d\n",i,label);
        if(write) 
        {
            if(splits.size()>0)
            {  
                if(splits[splitpos-1].y!=0 && splits[splitpos-1].y!=-99)
                    Y.push_back(splits[splitpos-1].y);
                else
                    Y.push_back(label);
            }
            else
                Y.push_back(label);
        }
			
        while(1)
        {
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            fscanf(fp,"%d:%lf",&index,&value);
			
            if (write==1) lasvm_sparsevector_set(X[m+i],index,value);
            if (write==2) lasvm_sparsevector_set(X[m+splitpos-1],index,value); // <-- was this the (and the only) bug?
            if (index>max_index) max_index=index;
        }
    out2:
        label=1; // dummy
    }

    fclose(fp);

    msz=X.size()-m;
    printf("examples: %d   features: %d\n",msz,max_index);

    return msz;
}

int binary_load_data(char *filename)
{
    int msz,i=0,j;
    lasvm_sparsevector_t* v;
    int nonsparse=0;

    ifstream f;
    f.open(filename,ios::in|ios::binary);
    
    // read number of examples and number of features
    int sz[2]; 
    f.read((char*)sz,2*sizeof(int));
    if (!f) { printf("File writing error in line %d of %s.\n",i,filename); exit(1);}
    msz=sz[0]; max_index=sz[1];

    vector <float> val;
    vector <int>   ind;
    val.resize(max_index);
    if(max_index>0) nonsparse=1;
    int splitpos=0;
	     
    for(i=0;i<msz;i++) 
    {
        int mwrite=0;
        if(splits.size()>0)
        {
            if(splitpos<(int)splits.size() && splits[splitpos].x==i) 
            { 
                mwrite=1;splitpos++;
                v=lasvm_sparsevector_create(); X.push_back(v);
            }
        }
        else
        { 
            mwrite=1;
            v=lasvm_sparsevector_create(); X.push_back(v);
        }
		
        if(nonsparse) // non-sparse binary file
        {
            f.read((char*)sz,1*sizeof(int)); // get label
            if(mwrite) 
            {
                if(splits.size()>0 && (splits[splitpos-1].y!=0 && splits[splitpos-1].y!=-99))
                    Y.push_back(splits[splitpos-1].y);
                else
                    Y.push_back(sz[0]);
            }
            f.read((char*)(&val[0]),max_index*sizeof(float));
            if(mwrite)
                for(j=0;j<max_index;j++) // set features for each example
                    lasvm_sparsevector_set(v,j,val[j]);
        }
        else			// sparse binary file
        {
            f.read((char*)sz,2*sizeof(int)); // get label & sparsity of example i
            if(mwrite) 
            {
                if(splits.size()>0 && (splits[splitpos-1].y!=0 && splits[splitpos-1].y!=-99))
                    Y.push_back(splits[splitpos-1].y);
                else 
                    Y.push_back(sz[0]);
            }
            val.resize(sz[1]); ind.resize(sz[1]);
            f.read((char*)(&ind[0]),sz[1]*sizeof(int));
            f.read((char*)(&val[0]),sz[1]*sizeof(float));
            if(mwrite)
                for(j=0;j<sz[1];j++) // set features for each example
                {
                    lasvm_sparsevector_set(v,ind[j],val[j]);
                    //printf("%d=%g\n",ind[j],val[j]);
                    if(ind[j]>max_index) max_index=ind[j];
                }
        }		
    }
    f.close();

    msz=X.size()-m;
    printf("examples: %d   features: %d\n",msz,max_index);

    return msz;
}


void load_data_file(char *filename)
{
    int msz,i,ft;
    splits.resize(0); 

    int bin=binary_files;
    if(bin==0) // if ascii, check if it isn't a split file..
    {
        FILE *f=fopen(filename,"r");
        if(f == NULL)
        {
            fprintf(stderr,"Can't open input file \"%s\"\n",filename);
            exit(1);
        }
        char c; fscanf(f,"%c",&c); 
        if(c=='f') bin=2; // found split file!
    }

    switch(bin)  // load diferent file formats
    {
    case 0: // libsvm format
        msz=libsvm_load_data(filename); break;
    case 1: 
        msz=binary_load_data(filename); break;
    case 2:
        ft=split_file_load(filename);
        if(ft==0) 	 
        {msz=libsvm_load_data(filename); break;} 
        else
        {msz=binary_load_data(filename); break;}
    default:
        fprintf(stderr,"Illegal file type '-B %d'\n",bin);
        exit(1);
    }

    if(kernel_type==RBF)
    {
        x_square.resize(m+msz);
        for(i=0;i<msz;i++){
            x_square[i+m]=lasvm_sparsevector_dot_product(X[i+m],X[i+m]);
	}
    }

    if(kgamma==-1)
        kgamma=1.0/ ((double) max_index); // same default as LIBSVM

    m+=msz;
    
}


int sample_type(int i){
    switch (Y[i]){
    case -3: 
	return UNLABSAMP;
	break;
    case -2:
	return UNIVESAMP;
	break;
    case -4:
	return EXTRASAMP;
	break;
    default:
	return TRAINSAMP;
    }
}


void load_data(char* input_file_name,char* universum_file_name,char* testset_file_name,char* unlabeled_file_name){

    printf("Loading data...\n\n");
    //load training data
    int m_old = 0;

    printf("Training data: \n");
    load_data_file(input_file_name);
	printf("Done!\n");

    // check, if data contains universum/unlabeled points 
    for (int i = m_old; i != m;++i){
      if (Y[i] >= -1) data_map[TRAIN].push_back(i);
      else if (Y[i] == -2) data_map[UNIVERSUM].push_back(i); // universum has -2 
      else if (Y[i] == -3) data_map[UNLABELED].push_back(i);  // unlabeled has -3
    }
    m_old = m;
    printf("\n");

    //load universum data
    printf("Universum: \n");
    if (universum_file_name[0] != '\0'){
      load_data_file(universum_file_name);
      for(int i=m_old;i!=m;++i){
	Y[i]=-2;
	data_map[UNIVERSUM].push_back(i);
      }
    }
    else  printf("No Universum specified!\n");
    m_old = m;
    printf("\n");
    

    //load test data
    printf("Test Data: \n");
    if (testset_file_name[0] != '\0'){
       load_data_file(testset_file_name);
	   printf("Done!\n");
       for(int i=m_old;i!=m;++i){
        data_map[TEST].push_back(i);
       }
    }else  printf("No test set specified!\n");
    m_old = m;
    printf("\n");



    printf("Unlabeled Data: \n");
    if (unlabeled_file_name[0] != '\0'){
       load_data_file(unlabeled_file_name);
 	   printf("Done!\n");
       for(int i=m_old;i != m;++i){
        Y[i] =-3;
        data_map[UNLABELED].push_back(i);
       }
    }else  printf("No unlabeled data specified!\n");

    // if C for unlabeled = 0, it should not use it
    if (C3 == 0.0) data_map[UNLABELED].resize(0); // KEEP?

    printf("\n");
    printf("\n\nData successfully loaded:\n \tNumber of training examples: %i\n \tNumber of test examples: %i\n",data_map[TRAIN].size(),data_map[TEST].size());
    printf("\tNumber of unlabeled examples: %i\n \tNumber of examples in universum: %i\n\n",data_map[UNLABELED].size(),data_map[UNIVERSUM].size()); // uncomment in future version
}



void load_model_param(const char *model_file_name){
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL){
	  fprintf(stderr,"Model file does not exist!\n");
	  exit(1);
	}
	double alpha0 = 0.0;
	int num_sv = 0;
	// read parameters
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"b0")==0){
		  fscanf(fp,"%lf",&b0);
		  printf("\tb0: %g\n",b0);
		}else if(strcmp(cmd,"kernel_type")==0){		
		  fscanf(fp,"%80s",cmd);
		  int i = 0;
		  for(i=0;kernel_type_table[i];++i){
		    if(strcmp(kernel_type_table[i],cmd)==0){
		      kernel_type=i;
		      break;
		    }
		  }
		  if (i == sizeof(kernel_type_table)){
		    fprintf(stderr,"kernel_type not know\n"); exit(1);
		  }else{
		    printf("\tkernel_type: %s\n",kernel_type_table[kernel_type]);
		  }
		  
			
		}else if(strcmp(cmd,"degree")==0){
		  fscanf(fp,"%lf",&degree);
		  printf("\tdegree: %g\n",degree);
		}else if(strcmp(cmd,"kgamma")==0){
		  fscanf(fp,"%lf",&kgamma);
		  printf("\tkgamma: %g\n",kgamma);
		}else if(strcmp(cmd,"coef0")==0){
			fscanf(fp,"%lf",&coef0);
			printf("\tcoef0: %g\n",coef0);
		}else if(strcmp(cmd,"alpha0")==0){
			fscanf(fp,"%lf",&alpha0);
			printf("\talpha0: %g\n",alpha0);
		}else if(strcmp(cmd,"total_sv")==0){
			fscanf(fp,"%d",&num_sv);
			printf("\tNumber of SVs: %d\n",num_sv);
		}else if(strcmp(cmd,"SV")==0){
		  while(1){
		      int c = getc(fp);
		      if(c==EOF || c=='\n') break;	
		  }
		  break;
		}else{
		  fprintf(stderr,"unknown text in model file\n");
		  exit(1);
		}
	  }
	fclose(fp);
}




int load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL){
	  fprintf(stderr,"Model file does not exist!\n");
	  exit(1);
	}
	double alpha0 = 0.0;
	int num_sv = 0;
	// read parameters
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"b0")==0){
		  fscanf(fp,"%lf",&b0);
		  printf("\tb0: %g\n",b0);
		}else if(strcmp(cmd,"kernel_type")==0){		
		  fscanf(fp,"%80s",cmd);
		  int i = 0;
		  for(i=0;kernel_type_table[i];++i){
		    if(strcmp(kernel_type_table[i],cmd)==0){
		      kernel_type=i;
		      break;
		    }
		  }
		  if (i == sizeof(kernel_type_table)){
		    fprintf(stderr,"kernel_type not know\n"); exit(1);
		  }else{
		    printf("\tkernel_type: %s\n",kernel_type_table[kernel_type]);
		  }
		  
			
		}else if(strcmp(cmd,"degree")==0){
		  fscanf(fp,"%lf",&degree);
		  printf("\tdegree: %g\n",degree);
		}else if(strcmp(cmd,"kgamma")==0){
		  fscanf(fp,"%lf",&kgamma);
		  printf("\tkgamma: %g\n",kgamma);
		}else if(strcmp(cmd,"coef0")==0){
			fscanf(fp,"%lf",&coef0);
			printf("\tcoef0: %g\n",coef0);
		}else if(strcmp(cmd,"alpha0")==0){
			fscanf(fp,"%lf",&alpha0);
			printf("\talpha0: %g\n",alpha0);
		}else if(strcmp(cmd,"total_sv")==0){
			fscanf(fp,"%d",&num_sv);
			printf("\tNumber of SVs: %d\n",num_sv);
		}else if(strcmp(cmd,"SV")==0){
		  while(1){
		      int c = getc(fp);
		      if(c==EOF || c=='\n') break;	
		  }
		  break;
		}else{
		  fprintf(stderr,"unknown text in model file\n");
		  exit(1);
		}
	  }

	// read sv_coef and SV

	int index; double value;
	int elements = 0;
	int msz = 0; int i =0;
	long pos = ftell(fp);
	lasvm_sparsevector_t* v;
	elements = 0;
	while(1){
	  int c = fgetc(fp);
	  switch(c){
	  case '\n':
	    v=lasvm_sparsevector_create();
	    X.push_back(v); 
            ++msz;
            elements=0;
            break;
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
 out:
    fseek(fp,pos,SEEK_SET);
    max_index = 0;
    for(i=0;i<msz;i++){
      double tmp_alpha;
      fscanf(fp,"%lf",&tmp_alpha);
      alpha.push_back(tmp_alpha);
      while(1){
	int c;
	do {
	  c = getc(fp);
	      if(c=='\n') goto out2;
	} while(isspace(c));
	ungetc(c,fp);
	fscanf(fp,"%d:%lf",&index,&value);
	lasvm_sparsevector_set(X[m+i],index,value);
	if (index>max_index) max_index=index;
      }
out2:
      if (tmp_alpha != 0) 
	data_map[SV].push_back(m+i);
      else
	data_map[UNLABELED].push_back(m+i);
      //tmp_alpha=1.0; // dummy
    }
    
    fclose(fp);
    if (data_map[UNLABELED].size() > 0) alpha.push_back(alpha0);
    msz=X.size()-m;
    printf("examples: %d   features: %d\n",msz,max_index);
    printf("|SV|: %d   |unlabeled|: %d\n",data_map[SV].size(),data_map[UNLABELED].size());
    return msz;
   

}


//******************* Kernel functions ***********************

double kernel(int i, int j, void *kparam)
{
  double dot;
 
  if (use_ext_K && (i == (int) X.size() || j == (int) X.size())){
      return global_ext_K[min(i, j )];
  }

  dot=lasvm_sparsevector_dot_product(X[i],X[j]);
  // sparse, linear kernel
  switch(kernel_type)
  {
      case LINEAR:
	  break;
      case POLY:
	  dot= pow(kgamma*dot+coef0,degree); break;
      case RBF:
	  dot= exp(-kgamma*(x_square[i]+x_square[j]-2*dot)); break;
      case SIGMOID:
	  dot=tanh(kgamma*dot+coef0); break;
      case CUSTOM:
	  dot=lasvm_sparsevector_get(X[i],j+1);  break;
  }
  kcalcs++;

  if (i==j && ker_ridge!=0){
    switch(sample_type(i)){
    case UNIVESAMP:
      dot += 1/C2;
      break;

    case UNLABSAMP:
      dot += 1/C3;
      break;

    case TRAINSAMP:
      dot += 1/C;
      break;
    }
  }

  return dot;
}


void compute_extra_K(){
	printf("Computing kernel values for extra column ...");
	global_ext_K.resize(X.size()+1);


	for (int i = 0; i !=(int) X.size()+1; ++i) global_ext_K[i] = 0.0; // initialize with zero
	
	if (C_star == 0.0 and !justtest){
	    printf(" [OK]\n");
	    return; // if the alpha for balancing is set to 0, there's nothing to do
     	}
	for (int i = 0; i !=(int) X.size(); ++i){
	    for (int j = 0; j != (int) data_map[UNLABELED].size(); ++j){
		global_ext_K[i] += kernel(i,data_map[UNLABELED][j],NULL);
	    }
	    global_ext_K[i] /= (double)data_map[UNLABELED].size();
	}


	for (int i = 0; i !=(int) data_map[UNLABELED].size(); ++i){
		global_ext_K[X.size()] += global_ext_K[data_map[UNLABELED][i]];
	}
	global_ext_K[X.size()] /= (double)data_map[UNLABELED].size();
	printf(" [OK]\n");
}



int mt1,mt2;
FILE *fp;



/************************* Testing routines ******************************************/

double calc_multi_class_accuracy(vector< vector <double> > *muYest){
    int pred, acc; double max;
    acc = 0;
    for (int j = 0; j != (int) data_map[TEST].size(); ++j){
	max =(*muYest)[0][j]; pred = labels[0];
	for (int i = 1; i != do_multi_class; ++i){
	    if ((*muYest)[i][j] > max){
		max = (*muYest)[i][j];
		pred = labels[i];
	    }
	    
      	}
	if (Y[data_map[TEST][j]] == pred) ++acc;
    }
    printf("===========================================\n");
    printf("   Accuracy= %g (%d/%d)\n",(acc/(double)data_map[TEST].size())*100,((int)acc),data_map[TEST].size());
    printf("===========================================\n");
    return 100 * (double)acc/(double)data_map[TEST].size();

}
  
 
double test_current_model(char *fname){
    int i,j; double y,balconstr; double acc=0;
    // don't add the ridge while testing on training data
    int ker_ridge_tmp; ker_ridge_tmp =ker_ridge;
    ker_ridge = 0;
    // clear vector that stores test results
    Yest.resize(0);
    D.resize(D.size()+1);
    for(i=0;i!=(int)data_map[TEST].size();++i){
 	y=b0; balconstr = 0.0;
	// calc kernel between the test and the mean of all unlabeled examples
        if (data_map[UNLABELED].size() > 0){
          balconstr += global_ext_K[data_map[TEST][i]];
	  // add to decision function with unconstraint alpha
          y+=alpha[m]*balconstr;// alpha_0*K_i0 + b
        }
	// calc remaining part of decision function
        for(j=0;j<m;++j){
	    if (alpha[j]==0) continue;
	    y+=alpha[j]*kernel(data_map[TEST][i],j,NULL);
	}
	// store function value
	Yest.push_back(y);
        if(y>=0) y=1; else y=-1;
	if(((int)y)==Y[data_map[TEST][i]]) acc++;
    }
    // store the function values for the report file
    for (int i = 0; i !=(int) Yest.size(); ++i)
	D[D.size()-1].push_back(Yest[i]);


    if (do_multi_class == 0){
	printf("===========================================\n");
	printf("   Accuracy= %g (%d/%d)\n",(acc/(double)data_map[TEST].size())*100,((int)acc),data_map[TEST].size());
	printf("===========================================\n");
    }
    
    // set back ker_ridge
    ker_ridge = ker_ridge_tmp;
    return (acc/(double)data_map[TEST].size())*100;
}


double test_current_multi_class_model(char *fname){

    vector< vector <double> > multi_Yest; multi_Yest.resize(do_multi_class);
    for (int i = 0; i != do_multi_class; ++i){
	alpha = multi_alpha[i]; b0 = multi_b0[i];
	test_current_model(fname);
	for (int j = 0; j != (int) data_map[TEST].size(); ++j){
	    multi_Yest[i].push_back(Yest[j]); 
	}
    }
    if (!use_universum == MULTIUNI){
	return calc_multi_class_accuracy(&multi_Yest);
    }else{
	Yest.resize(0);
	int y = 0; int acc = 0;
	D.resize(D.size()+1);
	
	for (int i = 0; i !=(int) data_map[TEST].size(); ++i){
	    Yest.push_back(multi_Yest[0][i] *(double)labels[0] + multi_Yest[1][i]*(double)labels[1]);
	    if (Yest[i] >= 0) y = 1; else y = -1;
	    if (y == Y[data_map[TEST][i]]) ++acc;
	}
	for (int i = 0; i !=(int) Yest.size(); ++i)
	    D[D.size()-1].push_back(Yest[i]);
	
	
	printf("===========================================\n");
	printf("   Accuracy= %g (%d/%d)\n",(acc/(double)data_map[TEST].size())*100,((int)acc),data_map[TEST].size());
	printf("===========================================\n");

	return  (acc/(double)data_map[TEST].size())*100;
    }
    
    
}


/*************************** postprocessing routines ************************************/

vector<int> get_unswap_permutation(SVQP2* sv){
    vector<int> retval(sv->n);
    for (int i=0; i< sv->n; i++)
	retval[sv->pivot[i]] = i;
    return retval;
}


void set_alphas_b0(SVQP2* sv){
    alpha.resize(0);
    b0 = 0;
    if (data_map[UNLABELED].size() > 0){ // add extra column alpha
     alpha.resize(m+1); // alpha of proportion of label constraint
//     alpha[m]+=sv->x[prob_size-1]; // fill in special alpha
    }else{
     alpha.resize(m); // keeps as many alphas as vectors
    }


    // set all alphas to zero
    for(int i=0;i<(int)alpha.size();i++){ alpha[i]=0;}
    // only training vectors may have non-zero alpha
    for(int i=0;i< sv->n;i++){
       alpha[sv->Aperm[i]]+=sv->x[i]; // fill in learnt svs
       if(sv->x[i]>maxa) maxa=(sv->x[i]);
       if(-(sv->x[i])>maxa) maxa=-(sv->x[i]);
    }
    if (optimizer == SVQP){
      b0=(sv->gmin+ sv->gmax)/2.0;
    }


    //printf("Maximum alpha=%g\n",maxa);
	
}

/************************* preprocessing routines*********************************************/

void setup_standard_svm_problem(SVQP2* sv){
    fill_level = 0;
    (*sv).verbosity=verbosity;
    (*sv).Afunction=kernel;
    (*sv).Aclosure=(void*) &kparam;
    (*sv).maxcachesize=(long int) 1024*1024*cache_size;
    (*sv).sumflag=use_b0;
    (*sv).epsgr=epsgr;
    for(int j=0;j<(int)data_map[TRAIN].size();++j){
	sv->Aperm[fill_level] = data_map[TRAIN][j];
	if (Y[data_map[TRAIN][j]] <0){
	    sv->b[fill_level]= -1;
	    sv->cmin[fill_level]=-C; sv->cmax[fill_level]=0;
	}else{
	    sv->b[fill_level]= 1;
	    sv->cmin[fill_level]=0; sv->cmax[fill_level]=C;
	}
	fill_level++;
    }
}


void extend_to_universum_problem(SVQP2* sv){

    double switchoff = 1.0-(double)(use_universum==RAMPUNI);// don't update first loop for ramp universum prob.
    for(int j=0;j<(int)data_map[UNIVERSUM].size();++j){
      // negative label universum point
      sv->Aperm[fill_level] = data_map[UNIVERSUM][j];
      sv->b[fill_level]= -1 * -gap;
      sv->cmin[fill_level]=-C2*switchoff; sv->cmax[fill_level]=0;
      fill_level++;

      // positive label universum point
      sv->Aperm[fill_level] = data_map[UNIVERSUM][j];
      sv->b[fill_level]= 1 * -gap;
      sv->cmin[fill_level]=0; sv->cmax[fill_level]=C2*switchoff;
      fill_level++;
    }
}

void setup_ramp_svm_problem(SVQP2* sv,bool firstloop){

    if (firstloop){
      printf("Setting up standard SVM problem...");
      setup_standard_svm_problem(sv);
    }else{
      vector<int> ip = get_unswap_permutation(sv);
      printf("Updating ramp loss SVM problem...");
      
      for(int j=0; j != sv->n;++j){
	  if (sample_type(sv->Aperm[ip[j]]) == TRAINSAMP){
	      if (Y[sv->Aperm[ip[j]]] == -1){  // negative label
		  sv->cmin[ip[j]]=-C -beta[j]; sv->cmax[ip[j]]=-beta[j]; // initialized to zero
	      }else{	// positive label
		  sv->cmin[ip[j]]=-beta[j]; sv->cmax[ip[j]]=C  -beta[j]; // initialized to zero
	      }
	  }
      }
    }
    printf("\t[OK]\n");
}

void extend_to_tsvm_problem(SVQP2* sv,bool firstloop){
    if (firstloop){
      printf("Setting up transductive SVM problem...");
      // add unlabeled points twice to problem; order - + - + ... 

      for(int j=0;j<(int)data_map[UNLABELED].size();++j){
	sv->Aperm[fill_level] = data_map[UNLABELED][j];
        sv->b[fill_level]= -1;
        sv->cmin[fill_level]=0; sv->cmax[fill_level]=0; // initialized to zero
        fill_level++;

	sv->Aperm[fill_level] = data_map[UNLABELED][j];
        sv->b[fill_level]= 1;
	sv->cmin[fill_level]=0; sv->cmax[fill_level]=0; // initialized to zero
        fill_level++;
      }

     
       // setup conditions for last special transductive column
      sv->cmin[sv->n -1] = -C_star;
      sv->cmax[sv->n -1] = C_star;
      sv->b[sv->n -1] = un_weight;
      sv->Aperm[sv->n -1] = X.size();
      Y.push_back(-4);

    }else{
      vector<int> ip = get_unswap_permutation(sv);
      printf("Updating ramp loss transductive SVM problem...");
      bool negsamp = 1;
      // change box constraint for new optimization problem with changed variables
      for(int j=0;j<prob_size-1;++j){
	  if (sample_type(sv->Aperm[ip[j]]) == UNLABSAMP){
	      if (negsamp){
                  // negative label
		  sv->cmin[ip[j]]=-C3-beta[j]; sv->cmax[ip[j]]=-beta[j]; // bounds for negative label unlabeled point
		  negsamp = 0;
	      }else{
		  // positive label
		  sv->cmin[ip[j]]=-beta[j]; sv->cmax[ip[j]]=C3-beta[j]; // bounds for positive labeled unlabeled point
		  negsamp = 1;
	      }
	  }
      }
    }
    printf("\t[OK]\n");

}


void extend_to_ramp_universum_problem(SVQP2* sv,bool firstloop){

    if (firstloop){
      printf("Extending to SVM + universum problem...");
      extend_to_universum_problem(sv);
    }else{
      printf("Updating ramp loss SVM universum...");
      // change box constraint for new optimization problem with changed variables
      bool negsamp = 1;
      vector<int> ip = get_unswap_permutation(sv);

      for(int j=0; j <  prob_size;++j){
	  if (sample_type(sv->Aperm[ip[j]]) == UNIVESAMP){
	      if (negsamp){
		  // negative label
		  sv->cmin[ip[j]]=-C2-beta[j]; sv->cmax[ip[j]]=-beta[j]; // update bounds for negative label universum point
		  negsamp = 0;
	      }else{
		  // positive label
		  sv->cmin[ip[j]]=-beta[j]; sv->cmax[ip[j]]=C2-beta[j]; // update bounds for positive labeled universum point
		  negsamp = 1;
	      }
	  }
      }
    }
    printf("\t[OK]\n");
}

/************************* optimization **********************************/

double do_ramp_loop(SVQP2* sv,bool firstiter){
    
  double beta_diff = 0.0; double y;
  int extra = (int)(data_map[UNLABELED].size() > 0);
  double uniLabel = 0.0;
  int sparsity = 0;
  bool neg_uni = 1; bool neg_unlab = 1;

  sv->run(firstiter,false); // first run?, last run = false
  vector<int> ip = get_unswap_permutation(sv);

 
  printf("Updating w and b ... ");

  vector<double> fvals_without_b;
  // Cache kernel expansions for beta update
  for(int i=0;i<prob_size-extra;i++){
      if (sv->x[ip[i]] != 0.0) ++sparsity;
      fvals_without_b.push_back(sv->b[ip[i]] - sv->g[ip[i]]); // y = f(x) - b
  }
  // compute b(t+1) 
  b0 = (sv->gmin+ sv->gmax)/2.0;
 

  // compute beta(t+1) for training point but only if we are not doing any transduction
  for(int i=0;i<prob_size-extra;i++){
      y = fvals_without_b[i] + b0; // y = f(x)
      if (optimizer == CCCP){ // only when CCCP is explicitly specified for TRAIN and UNIVERSUM
	  if (sample_type(sv->Aperm[ip[i]]) == TRAINSAMP && use_universum != RAMPUNI){
	      // determine new beta[i](t+1)
	      if (Y[sv->Aperm[ip[i]]]*y<s_ramp){
		  beta_diff += (beta[i]-Y[sv->Aperm[ip[i]]]*C)*(beta[i]-Y[sv->Aperm[ip[i]]]*C);
		  beta[i] = Y[sv->Aperm[ip[i]]]*C;
	      }else{
		  beta_diff += (beta[i])*(beta[i]);
		  beta[i] = 0.0;
	      }
	  }else if (sample_type(sv->Aperm[ip[i]]) == UNIVESAMP){
	      // compute beta(t+1) for universum point
	      if (neg_uni){ // negative label
		  uniLabel = -1.0;
	      }else{ // positive label
		  uniLabel = 1.0;
	      }
	      neg_uni = !neg_uni;
	      
	      if (uniLabel*y<s_ramp-gap-1){
		  beta_diff += (beta[i]- uniLabel*C2)*(beta[i]- uniLabel*C2);
		  beta[i] = uniLabel*C2;
	      }else{
		  beta_diff += (beta[i])*(beta[i]);
		  beta[i] = 0.0;
	      }
	  }
      }

      // update beta for unlabeled examples
      if (sample_type(sv->Aperm[ip[i]]) == UNLABSAMP){
	  if (neg_unlab){ // negative label
	      uniLabel = -1.0;
	  }else{ // positive label
	      uniLabel = 1.0;
	  }
	  neg_unlab = !neg_unlab;
	  if (uniLabel*y<s_trans){
	      beta_diff += (beta[i]-uniLabel*C3)*(beta[i]-uniLabel*C3);
	      beta[i] = uniLabel*C3;
	  }else{
	      beta_diff += (beta[i])*(beta[i]);
	      beta[i] = 0.0;
	  }
      }
  }


  printf("\t[OK]\n");
  printf("Sparsity=(%i/%i)\n",sparsity,prob_size-extra);

  double retval =  beta_diff/(double)prob_size;
  if (retval == 0.0) sv->run(false,true); //cleanup hack
  return retval; //return mean quadratic error

}


void reset_alphas(SVQP2* sv){
    for (int i = 0; i != sv->n;++i){
	sv->x[i] = 0.0;
    }
}

/********************************** Saving Functions ************************************/

void print_fvals(char* fval_file_name){
  FILE *fp2;
  fp2 = fopen(fval_file_name,"w");
  fprintf(fp2,"\n# function values\n");
  for (int i = 0; i !=(int) D.size(); ++i){
      fprintf(fp2,"f{%i} = [ %g",i+1,D[i][0]);
      for (int j = 1; j != (int) D[i].size(); ++j)
	  fprintf(fp2," ,%g",D[i][j]);
      fprintf(fp2,"];\n");
  }
  fclose(fp2);
}

void print_model_file(char* model_file_name){
  FILE *fp2;
  fp2 = fopen(model_file_name,"w");

  fprintf(fp2,"b0 %g\n",b0);
  fprintf(fp2,"kernel_type %s\n",kernel_type_table[kernel_type]);
  fprintf(fp2,"kgamma %.15g\n",kgamma);
  fprintf(fp2,"degree %g\n",degree);
  fprintf(fp2,"coef0 %g\n",coef0);
  if (alpha.size() == m+1)// write out balancing constraint alpha
    fprintf(fp2,"alpha0 %.15g\n",alpha[m]);
  int num_sv = 0;
  for(int i=0;i<m;++i){
    if (alpha[i]!=0) ++num_sv;
  }    
  fprintf(fp2,"total_sv %d\n",num_sv);
  fprintf(fp2,"SV\n");
  // write out SVs
  for(int i=0;i<m;++i){
    if (alpha[i]==0) continue;
    fprintf(fp2,"%.15g ",alpha[i]);
    lasvm_sparsevector_pair_t *p;
    for(p=X[i]->pairs; p->index>=0; p++) {
      fprintf(fp2," %d:%.15g ", p->index, p->value);
    }
    fprintf(fp2,"\n");
    
  }
  // write out unlabeled data
  for(int i=0;i<data_map[UNLABELED].size();++i){
    fprintf(fp2,"0 ");
    lasvm_sparsevector_pair_t *p;
    for(p=X[data_map[UNLABELED][i]]->pairs; p->index>=0; p++) {
      fprintf(fp2," %d:%.8g ", p->index, p->value);
    }
    fprintf(fp2,"\n");
    
  }

  fclose(fp2);
  
}


void print_report(vector<double> testerr){
  FILE *fp2;
  fp2 = fopen(report_file_name,"w");
  fprintf(fp2,"# data set sizes\n");
  fprintf(fp2,"training=%i\n",data_map[TRAIN].size());
  fprintf(fp2,"test=%i\n",data_map[TEST].size());
  fprintf(fp2,"unlabeled=%i\n",data_map[UNLABELED].size());
  //fprintf(fp2,"universum=%i\n",data_map[UNIVERSUM].size());

  if (folds == -1){
    fprintf(fp2,"\n# training parameters\n");
  }else{
    fprintf(fp2,"\n# %i-fold cross validation parameters\n",folds);
  }
  fprintf(fp2,"c=%g\n",C);
  fprintf(fp2,"C=%g\n",C2);
  fprintf(fp2,"z=%g\n",C3);
  fprintf(fp2,"t=%i\n",kernel_type);
  if (optimizer == CCCP) fprintf(fp2,"s=%g\n",s_ramp);
  fprintf(fp2,"G=%g\n",gap);

  if (data_map[UNLABELED].size() > 0){
      fprintf(fp2,"S=%g\n",s_trans);
      fprintf(fp2,"<y>_train=%g\n",un_weight);
  }

  switch (kernel_type){
    case RBF:
      fprintf(fp2,"kernel=RBF\n");
      fprintf(fp2,"g=%g\n",kgamma);
      break;
    case SIGMOID:
      fprintf(fp2,"kernel=SIGMOID\n");
      fprintf(fp2,"r=%g\n",coef0);
      break;
    case POLY:
      fprintf(fp2,"kernel=POLY\n");
      fprintf(fp2,"r=%g\n",coef0);
      fprintf(fp2,"d=%g\n",degree);
      break;
    default:
      fprintf(fp2,"kernel=LINEAR\n");
      break;
  }
  fprintf(fp2,"r=%g\n",coef0);

  fprintf(fp2,"\n# algorithm parameters\n");
  fprintf(fp2,"o=%i\n",optimizer);
  fprintf(fp2,"optimizer=%s\n",optimizer_table[optimizer]);

  fprintf(fp2,"Max_alpha=%g\n",maxa);

  fprintf(fp2,"\n# training times\n");
  for (int i = 0; i !=(int) training_time.size(); ++i){
    fprintf(fp2,"time %i =%g\n",i+1,training_time[i]);
  }



  fprintf(fp2,"\n# accuracies\n");
  for (int i = 0; i !=(int) testerr.size()-1; ++i){
    fprintf(fp2,"fold %i =%g\n",i+1,testerr[i]);
  }
  fprintf(fp2,"mean accuracy=%g\n",testerr[testerr.size()-1]);


  fclose(fp2);
}


/********************************************************************************************/


void setup_problem(SVQP2* sv){
    if (optimizer == SVQP && data_map[UNLABELED].size() == 0){
        setup_standard_svm_problem(sv);
        if (data_map[UNIVERSUM].size() > 0){// train SVQP with universum?
          extend_to_universum_problem(sv);
        }
    }else if (optimizer == CCCP || data_map[UNLABELED].size() > 0){
      if (data_map[UNLABELED].size() > 0){ // train transductive svm?
        b0 = 0.0; 

	if (isnan(un_weight)) {
	  un_weight = 0.0;
	  for (int i = 0; i !=(int) data_map[TRAIN].size();++i){
	    un_weight += (double)Y[data_map[TRAIN][i]];
	  }
	  un_weight /= (double) data_map[TRAIN].size();
	}
       	if (optimizer == CCCP){
	    setup_ramp_svm_problem(sv,1);
	}else{
	    setup_standard_svm_problem(sv);
       	}
        beta.resize(0);
        for (int i = 0; i != prob_size; ++i){
          beta.push_back(0.0);
        }

        if (data_map[UNIVERSUM].size() > 0){// train transductive CCCP with universum?
          extend_to_ramp_universum_problem(sv,1);
        }

        // extend to unlabeled problem
        if (data_map[UNLABELED].size() > 0) extend_to_tsvm_problem(sv,1);
      }else{ // train non-transductive svm
        beta.resize(0);
        b0 = 0.0;
        // initialize loop variables
        for (int i = 0; i != prob_size; ++i){
          beta.push_back(0.0);
        }
        setup_ramp_svm_problem(sv,1);
        if (data_map[UNIVERSUM].size() > 0){// train CCCP with universum?
          extend_to_ramp_universum_problem(sv,1);
        }
      }
    }
}

void training(SVQP2* sv){
    printf("\nTraining ...\n");
    stopwatch* stop = new stopwatch(); double start_time;
    stop->reset();
    start_time = stop->get_time();

    if (data_map[UNLABELED].size() > 0) use_ext_K = 1;

    if (optimizer == SVQP && data_map[UNLABELED].size() == 0){
	  sv->run(true,true);
    }else if(optimizer == CCCP || data_map[UNLABELED].size() > 0){
      double beta_diff = 1.0;
      beta_diff = do_ramp_loop(sv,1); 
	  int numloops=0;
      while ((beta_diff > CCCP_TOL) && numloops<MAX_LOOPS)
	  {
	    
		numloops++;
		printf("\n---- Loop %d/%d ----\n",numloops,MAX_LOOPS);
		if (optimizer == CCCP)
		{
		    setup_ramp_svm_problem(sv,0);
			 if (data_map[UNIVERSUM].size() > 0)
			 {// train CCCP with universum?
				extend_to_ramp_universum_problem(sv,0);
			 }
		}
        if (data_map[UNLABELED].size() > 0)
		{// train transductive CCCP?
          extend_to_tsvm_problem(sv,0);
        }
		reset_alphas(sv);
        beta_diff = do_ramp_loop(sv,0);
      }
	  if(numloops==MAX_LOOPS) {printf("Max number of loops exceeded!\n");}
      use_ext_K = 0;
    }
    training_time.push_back(stop->get_time()-start_time);
    delete stop;
    printf("\nTraining done ...\n");

}

void search_for_different_labels(){
    labels.resize(0);
    bool found = 0;

    for (int i = 0; i !=(int) Y.size(); ++i)
    {
		for (int j = 0; j != (int) labels.size(); ++j)
		{
			if (Y[i] < -1 || Y[i] == labels[j]){ found = 1; break;}
		}
		if (!found && Y[i]>=-1) labels.push_back(Y[i]);
		if ((int) labels.size() == do_multi_class) break;
		found = 0;
    }

}

void setup_labels_for_multiclass(){
    printf("Setting up labels for multiclass ...");
    multi_Y.resize(do_multi_class);
    for (int j = 0; j != (int) Y.size(); ++j){
	for (int i = 0; i != do_multi_class; ++i){
	    if (Y[j] == labels[i]) multi_Y[i].push_back(1);
	    else if (Y[j] < 0) multi_Y[i].push_back(Y[j]);
	    else multi_Y[i].push_back(-1);
	}
    }
    printf("\t[OK]\n");
}

void multi_class_training(){
	  /*
	    Do single multi-class training and testing
	  */
	  vector<int> Ybck = Y;
	  SVQP2* sv;
	  for (int i = 0; i != do_multi_class; ++i){
	      sv = new SVQP2(prob_size);
	      Y = multi_Y[i];

	      setup_problem(sv);

	      printf("---------------------------------\n");
	      printf(" Training %i  vs. Rest\n",labels[i]);
	      printf(" Overall size of training set: %i\n",sv->n);
	      printf("---------------------------------\n");

	      training(sv);
	      // processing results
	      set_alphas_b0(sv);
	      multi_alpha[i].resize(0);
	      for (int j = 0; j != (int) alpha.size(); ++j){
		  multi_alpha[i].push_back(alpha[j]);
	      }
	      multi_b0.push_back(b0);
	      delete sv;

	  }
	  Y = Ybck;

}

void setup_folds(vector <  vector <int> >* a, vector <  vector <int> >* b){
  // store train and test indices in trainfolds and testfolds
  // generate folds ...
  // get training data
    
  vector < vector <int> > label_map; label_map.resize(labels.size());

  // get indices of positive and negative labels
  for (int i = 0; i !=(int) data_map[TRAIN].size(); ++i){
      for (int k = 0; k != (int) labels.size(); ++k){
	  if (Y[data_map[TRAIN][i]] == labels[k]){
	      label_map[k].push_back(data_map[TRAIN][i]);
	      break; 
	  }
      }
  }
  int balancecv=0;
  vector <int> props; props.resize(labels.size());
  int propsum = 0;

  for (int k = 0; k != (int) labels.size(); ++k){ // for each label

      props[k] = label_map[k].size()/folds;   // count how many we can put in each fold
      propsum += props[k];
      if (props[k] < 1){
	  printf("Warning: making sure each fold has enough training examples of each class, using subsampling instead.\n");
	  balancecv=1;
      }
  }



  (*a).resize(folds); (*b).resize(folds);
  vector <int> pointers; for (int k = 0; k != (int) labels.size(); ++k) pointers.push_back(0);


  if(!balancecv){
    for (int i = 0; i != folds-1; ++i){
      

      // get training vectors before testfold
      for (int k = 0; k != (int) labels.size(); ++k){
	for (int j = 0; j != (int) pointers[k]; ++j){
	  (*a)[i].push_back(label_map[k][j]);
	}
      }


      // get test fold
      for (int k = 0; k != (int) labels.size(); ++k){
	for (int j = pointers[k]; j != (int) pointers[k] + props[k]; ++j){
	  (*b)[i].push_back(label_map[k][j]);
	}
      }

      // update pointers
      for (int k = 0; k != (int) labels.size(); ++k){
	pointers[k] += props[k];
      }
    

      // get training vectors after testfold
      for (int k = 0; k != (int) labels.size(); ++k){
	for (int j = pointers[k]; j != (int) label_map[k].size(); ++j){
	  (*a)[i].push_back(label_map[k][j]);
	}
      }

    }


    // get training vectors before testfold
    for (int k = 0; k != (int) labels.size(); ++k){
      for (int j = 0; j != (int) pointers[k]; ++j){
	(*a)[folds-1].push_back(label_map[k][j]);
      }
    }
    
    // get the remaining stuff in test fold
    for (int k = 0; k != (int) labels.size(); ++k){
      for (int j = pointers[k]; j != (int) label_map[k].size(); ++j){
	(*b)[folds-1].push_back(label_map[k][j]);
      }
    }
    
  }else{// force to add a point from each class
    int i,j,k,p,q;
    //printf("jasons method on %d splits, %d labels %d %d\n",folds,labels.size(),labels[0],labels[1]);
    int labs=labels.size();
    vector <int> data;
    int train_m = data_map[TRAIN].size(); //train_m was =m before.
    for(i=0;i<folds;i++)
      {
	data.resize(0); for(j=0;j<train_m;j++) data.push_back(j);
	for(j=0;j<labs;j++) // add at least one example from each class
	  {
	    p= (int) ( ((float)i)/((float)folds) * ((float)data.size()));
	    for(k=0;k<train_m;k++)
	      {
		if(Y[(k+p)%train_m]==labels[j]) { q=(k+p)%train_m;break;} // find example of each class
	      }
	    (*a)[i].push_back(q); data[q]=data[data.size()-1]; data.pop_back();
	  }
	
	// now add some random examples to make roughly folds-1/folds of the dataset
	p= (int) ( ((float)i)/((float)folds) * ((float)data.size()));
	int max=(int) (((float)(folds-1.0))/((float)folds)*train_m);
	for(j=0;j<max-(*a)[i].size();j++) // add at least one example from each class
	  {
	    p=p% data.size();
	    (*a)[i].push_back(data[p]); 
	    data[p]=data[data.size()-1]; data.pop_back();
	    p++;
	  }
	
	for(j=0;j<(int)data.size();j++) // add rest of examples to test set
	  (*b)[i].push_back(data[j]);
	
      }
    
  }
  
  
}

#ifndef MEX
/*******************************************************************************************/
int main(int argc, char **argv)
{
    printf("=========================================================\n");
    printf("||                                                     ||\n");
    printf("||       UniverSVM                                     ||\n");
    printf("||                                                     ||\n");
    printf("||       SVM for large scale transduction with CCCP,   ||\n");
    printf("||       sparse training with CCCP and induction       ||\n");
    printf("||       with universum.                               ||\n");
    printf("||                                                     ||\n");
    printf("||       Version 1.1                                   ||\n");
    printf("||                                                     ||\n");
    printf("==========================================================\n");

    //char input_file_name[1024];
    char input_file_name[1024]; input_file_name[0] = '\0';
    char model_file_name[1024]; model_file_name[0] = '\0';
    char testset_file_name[1024]; testset_file_name[0] = '\0';
    char universum_file_name[1024]; universum_file_name[0] = '\0';
    char unlabeled_file_name[1024]; unlabeled_file_name[0] = '\0';
    char fval_file_name[1024]; fval_file_name[0] = '\0';
    vector<double> accus; D.resize(0);
    int training_set_size = 0; // remember training set size for special universum training

    parse_command_line(argc, argv, input_file_name, universum_file_name, testset_file_name,unlabeled_file_name,model_file_name, fval_file_name);
    // overwrite parameter configureation with the stored one
    if (justtest) load_model_param(model_file_name);

    /** Catch some errors **/
    if (use_universum == MULTIUNI && do_multi_class){
	printf("%i\n", use_universum);
	fprintf(stderr,"Error: Universum Training with multiclass does not support multiclass at the moment.\n");
	return 1;
    }
    /**------------------**/

    printf("---------------------------------\n");
    printf("         Initalization   \n");
    printf("---------------------------------\n");
    
    if(split_file_name[0]!='\0') split_file_load(split_file_name);
    load_data(input_file_name, universum_file_name, testset_file_name,unlabeled_file_name);
    search_for_different_labels(); // check how many different labels we have (universum and unlabeled excluded)
    printf("Data contains %i classes \n\n",labels.size());

    // set default value for z (=C3)
    if (C3 < 0)
      C3 = (((double)data_map[TRAIN].size())/((double)data_map[UNLABELED].size()))*C;

      
    if (!justtest){// if we train a model

      
      for (int i = 0; i != do_multi_class; ++i){
	printf(" Training %i  vs. Rest\n",labels[i]);
      }
      
      
      prob_size =  data_map[TRAIN].size() + data_map[UNIVERSUM].size()*2 + data_map[UNLABELED].size()*2;
      if (data_map[UNLABELED].size() > 0){
	++prob_size; // last special column for transductive constraint
	compute_extra_K();      // setup extended column
      }
      
      
      // train
      SVQP2* sv = new SVQP2(prob_size);
      
      // catch special training switch for universum
      if (use_universum == MULTIUNI){
	printf("Setting up labels for multiclass universum training ... ");
	training_set_size = data_map[TRAIN].size();
	prob_size -=  data_map[UNIVERSUM].size();
    	int newLab = max(labels[0],labels[1])+1;
	labels.push_back(newLab);
	for (int i = 0; i !=(int) data_map[UNIVERSUM].size();++i){
	  data_map[TRAIN].push_back(data_map[UNIVERSUM][i]);
	  Y[data_map[UNIVERSUM][i]] = newLab;
	}
	data_map[UNIVERSUM].resize(0);
	do_multi_class = 3;
	printf("  [OK]\n");
      }
      
      sv = new SVQP2(prob_size);
      
      if (folds == -1){ // no cv
	if (do_multi_class == 0){ 
	  printf("---------------------------------\n");
	  printf("           Training        \n");
	  printf("---------------------------------\n");
	  setup_problem(sv);
	  training(sv);
	  // processing results
	  set_alphas_b0(sv);
	  // test
	  printf("---------------------------------\n");
	  printf("           Testing        \n");
	  printf("---------------------------------\n");
	  if (data_map[TEST].size() > 0){
	    printf("Testing on test set with %i examples:\n",data_map[TEST].size());
	    accus.push_back(test_current_model(model_file_name));
	  }else{
	    
	    printf("Testing on training set with %i examples:\n",data_map[TRAIN].size());
	    data_map[TEST] = data_map[TRAIN];
	    accus.push_back(test_current_model(model_file_name));
	    ///test_current_model(model_file_name);
	  }
	  delete sv;
	  
	}else{
	  ///////////////// Start multi_class ///////////////////////////////
	  multi_alpha.resize(do_multi_class);
	  setup_labels_for_multiclass();
	  if (use_universum == MULTIUNI) --do_multi_class;
	  
	  
	  multi_class_training();
	  
	  printf("---------------------------------\n");
	  printf("           Testing        \n");
	  printf("---------------------------------\n");
	  if (data_map[TEST].size() > 0){
	    printf("Testing on test set with %i examples:\n",data_map[TEST].size());
	  }else{
	    if (use_universum == MULTIUNI) data_map[TRAIN].resize(training_set_size);
	    printf("Testing on training set with %i examples:\n",data_map[TRAIN].size());
	    data_map[TEST] = data_map[TRAIN];
	  }
	  accus.push_back(test_current_multi_class_model(model_file_name));
	}
	///////////////// End multi_class ///////////////////////////////
	
      }else{
	vector < vector <int> > trainfold, testfold;
	setup_folds(&trainfold, &testfold);
	double meanacc = 0.0;
	double foldacc;
	if (do_multi_class > 0) setup_labels_for_multiclass();
	
	for (int fold = 0; fold != folds; ++fold){
	  
	  data_map[TRAIN] = trainfold[fold];
	  data_map[TEST] = testfold[fold];
	  
	  prob_size =  data_map[TRAIN].size() + data_map[UNIVERSUM].size()*2 + data_map[UNLABELED].size()*2;
	  if (data_map[UNLABELED].size() > 0) ++prob_size; // last special column for transductive constraint
	  
	  printf("---------------------------------\n");
	  printf(" Training fold (%i/%i)\n",fold+1,folds);
	  printf(" Overall size of training set: %i\n",prob_size);
	  printf("---------------------------------\n");
	  
	  
	  if (do_multi_class == 0){
	    SVQP2* sv = new SVQP2(prob_size);
	    setup_problem(sv);
	    training(sv);
	    // processing results
	    set_alphas_b0(sv);
	    foldacc = test_current_model(model_file_name);
	    meanacc +=  foldacc;
	    accus.push_back(foldacc);
	    delete sv;
	  }else{
	    multi_alpha.resize(do_multi_class);
	    
	    multi_class_training();
	    
	    printf("---------------------------------\n");
	    printf("           Testing        \n");
	    printf("---------------------------------\n");
	    if (data_map[TEST].size() > 0){
	      printf("Testing on test set with %i examples:\n",data_map[TEST].size());
	    }else{
	      printf("Testing on training set with %i examples:\n",data_map[TRAIN].size());
	      data_map[TEST].resize(0);
	      for (int i = 0; i !=(int) data_map[TRAIN].size(); ++i) 
		data_map[TEST].push_back(data_map[TRAIN][i]);
	    }
	    foldacc = test_current_multi_class_model(model_file_name);
	    meanacc +=  foldacc;
	    accus.push_back(foldacc);
	  }
	  
	}
	meanacc /= double(folds);
	accus.push_back(meanacc);
	
	
	printf("===========================================\n");
	printf("\nMean Cross Validation Accuracy: %g\n", meanacc);
	printf("===========================================\n");
      }
      
      if (report_file_name[0] != '\0'){
	print_report(accus);
      }  
      
      
      if (model_file_name[0] != '\0'){
	print_model_file(model_file_name);
      }  
    }else{
	printf("Loading model file...\n");
	alpha.resize(0);
	for (int i = 0; i != m; ++i) alpha.push_back(0.0); // set all alpha's of nonSV to zero
	data_map[SV].resize(0); data_map[UNLABELED].resize(0);
	int msz = load_model(model_file_name);

	if(kernel_type==RBF){
	  x_square.resize(m+msz);
	  for(int i=0;i<msz;i++){
            x_square[i+m]=lasvm_sparsevector_dot_product(X[i+m],X[i+m]);
	  }
	}
	m+=msz;
	if (data_map[UNLABELED].size() > 0){
	  use_ext_K = 1;
	  compute_extra_K();
	}
	// we are testing on the data sepcified with -T and training
	for (int i = 0; i != data_map[TRAIN].size(); ++i){
	  data_map[TEST].push_back(data_map[TRAIN][i]);
	}
	test_current_model(model_file_name);
    }


    if (fval_file_name[0] != '\0'){
      print_fvals(fval_file_name);
    }  
    return 0;
}
#endif

#ifdef MEX
void mexFunction(
		 int nlhs,              // Number of left hand side (output) arguments
		 mxArray *plhs[],       // Array of left hand side arguments
		 int nrhs,              // Number of right hand side (input) arguments
		 const mxArray *prhs[]  // Array of right hand side arguments
		 )
{
  double *X_;
  double *Y_;
  char cmd_[1000];
  int argc;
  char *argv[100];
  char input_file_name[1024]; input_file_name[0] = '\0';
  char model_file_name[1024]; model_file_name[0] = '\0';
  vector<double> accus; D.resize(0);
  int training_set_size = 0; // remember training set size for special universum training

  if (nrhs!=3) 
    mexErrMsgTxt("3 input arguments needed.\n");
    
  if (nlhs!=2) 
    mexErrMsgTxt("2 output arguments needed.\n");

  m  = mxGetM (prhs[0]);
  max_index = mxGetN (prhs[0]);
  X_ = mxGetPr(prhs[0]);

  if (mxGetM(prhs[1]) != m)
    mexErrMsgTxt("Dimension error.\n");

  Y_ = mxGetPr(prhs[1]);

  mxGetString(prhs[2],cmd_,1000);

  argv[1] = cmd_;
  argc = 2;
  for (char *p=cmd_; *p != '\0'; p++) {
    if (*p == ' ') {
      *p = '\0';
      while (*(++p) == ' ');
      argv[argc++] = p;
    }
  }
  argv[argc++] = "dummy file";
 
  parse_command_line(argc, argv, input_file_name, NULL, NULL, NULL, NULL, NULL);

  lasvm_sparsevector_t* v;
  for (int i=0; i<m; i++) {
    v=lasvm_sparsevector_create();
    X.push_back(v);
    Y.push_back((int)Y_[i]);
    if (Y[i] >= -1) data_map[TRAIN].push_back(i);
    else if (Y[i] == -2) data_map[UNIVERSUM].push_back(i); // universum has -2 
    else if (Y[i] == -3) data_map[UNLABELED].push_back(i);  // unlabeled has -3
    for (int j=0; j<max_index; j++){
      if (X_[j*m+i] != 0.0){
	lasvm_sparsevector_set(X[i],j,X_[j*m+i]);      
      }
    }
  }

  if(kernel_type==RBF){
    x_square.resize(m);
    for(int i=0;i<m;i++){
      x_square[i]=lasvm_sparsevector_dot_product(X[i],X[i]);
    }
  }

  if(kgamma==-1)
    kgamma=1.0/ ((double) max_index); // same default as LIBSVM

  // set default value for C3 (=z = regularization cost for unlabeled examples)
  if (C3 < 0)
    C3 = (((double)data_map[TRAINING].size())/((double)data_map[UNLABELED].size()))*C;


  // the remaining part o the main function
  search_for_different_labels(); // check how many different labels we have (universum and unlabeled excluded)
  printf("Data contains %i classes \n\n",labels.size());
  for (int i = 0; i != do_multi_class; ++i){
    printf(" Training %i  vs. Rest\n",labels[i]);
  }
      
  prob_size =  data_map[TRAIN].size() + data_map[UNIVERSUM].size()*2 + data_map[UNLABELED].size()*2;
  if (data_map[UNLABELED].size() > 0){
    ++prob_size; // last special column for transductive constraint
    compute_extra_K();      // setup extended column
  }
      
      
  // train
  SVQP2* sv = new SVQP2(prob_size);
      
  // catch special training switch for universum
  if (use_universum == MULTIUNI){
    printf("Setting up labels for multiclass universum training ... ");
    training_set_size = data_map[TRAIN].size();
    prob_size -=  data_map[UNIVERSUM].size();
    int newLab = max(labels[0],labels[1])+1;
    labels.push_back(newLab);
    for (int i = 0; i !=(int) data_map[UNIVERSUM].size();++i){
      data_map[TRAIN].push_back(data_map[UNIVERSUM][i]);
      Y[data_map[UNIVERSUM][i]] = newLab;
    }
    data_map[UNIVERSUM].resize(0);
    do_multi_class = 3;
    printf("  [OK]\n");
  }
      
  sv = new SVQP2(prob_size);

  if (do_multi_class == 0){ 
    printf("---------------------------------\n");
    printf("           Training        \n");
    printf("---------------------------------\n");
    setup_problem(sv);
    training(sv);
    // processing results
    set_alphas_b0(sv);
    plhs[0] = mxCreateDoubleMatrix(m,1,mxREAL);
    double *p=mxGetPr(plhs[0]);
    for (int i=0; i<m; i++)
      p[i]= alpha[i];
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[1]) = b0;
    delete sv;
    
  }else{
    ///////////////// Start multi_class ///////////////////////////////
    multi_alpha.resize(do_multi_class);
    setup_labels_for_multiclass();
    if (use_universum == MULTIUNI) --do_multi_class;
    multi_class_training();
    plhs[0] = mxCreateDoubleMatrix(m,do_multi_class,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,do_multi_class,mxREAL);
    double *p1=mxGetPr(plhs[0]);
    double *p2=mxGetPr(plhs[1]);
    for (int i=0; i<do_multi_class; i++) {
      alpha = multi_alpha[i];
      for (int j=0; j<m; j++)
	*p1++ = alpha[j];
      *p2++ = multi_b0[i];
    }
  }
  ///////////////// End multi_class ///////////////////////////////

  for (int i=0; i<m; i++)
    lasvm_sparsevector_destroy(X[i]);
  X.clear();
  Y.clear();
  data_map[TRAIN].clear();
  data_map[UNLABELED].clear();
  data_map[UNIVERSUM].clear();
  data_map[SV].clear();
}
#endif
