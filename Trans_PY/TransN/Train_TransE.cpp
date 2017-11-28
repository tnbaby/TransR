#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
using namespace std;


#define pi 3.1415926535897932384626433832795

bool L1_flag=1;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

string version;
string data_path = "../../FB15k/";
char buf[100000],buf1[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;


map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;
class Train{

public:
	map<pair<int, int>, vector<int> >entity_neighbors;
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x,z)][y]=1;
	entity_neighbors[make_pair(x,z)].push_back(y);
//	if entity_neighbors.count(make_pair(x,z)) > 0
//		entity_pair[make_pair(x,z)].push_back(y);
//	else
//		entity_pair[make_pair(x,z)].push_back(y);
    }
    void run(int n_in,double rate_in,double margin_in,int method_in, double con_alpha_in)
    {
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
	con_alpha = con_alpha_in;
        relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
	for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }
	
        bfgs();
    }

private:
    int n,method;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate,margin, con_alpha;
    double belta;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<int> > feature;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<double> > relation_tmp,entity_tmp;
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
        res=0;
        int nbatches=100;
        int nepoch = 1000;
        int batchsize = fb_h.size()/nbatches;
            for (int epoch=0; epoch<nepoch; epoch++)
            {

            	res=0;
		time_t epoch_s, epoch_e;
		epoch_s = time(NULL);
             	for (int batch = 0; batch<nbatches; batch++)
             	{
			time_t start = time(NULL), stop;
			int tmp_res = res;
             		relation_tmp=relation_vec;
            		entity_tmp = entity_vec;
             		for (int k=0; k<batchsize; k++)
             		{
						int i=rand_max(fb_h.size());
						int j=rand_max(entity_num);
						double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
						if (method ==0)
			                            pr = 500;
						if (rand()%1000<pr)
						{
							while (ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
								j=rand_max(entity_num);
							train_kb(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i]);
						}
						else
						{
							while (ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
								j=rand_max(entity_num);
							train_kb(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i]);
						}
                		norm(relation_tmp[fb_r[i]]);
                		norm(entity_tmp[fb_h[i]]);
                		norm(entity_tmp[fb_l[i]]);
                		norm(entity_tmp[j]);
             		
			}
			stop = time(NULL);
			cout << "batch: " << batch << " res: " << res - tmp_res << " time" << (double)(stop-start) << endl;
		        relation_vec = relation_tmp;
		        entity_vec = entity_tmp;
             	}
		epoch_e = time(NULL);
                cout<<"epoch:"<<epoch<<' '<<res<<(double)(epoch_e-epoch_s)<<endl;
                FILE* f2 = fopen(("relation2vec."+version).c_str(),"w");
                FILE* f3 = fopen(("entity2vec."+version).c_str(),"w");
		if(f2 ==  NULL || f3 == NULL)
		   cout << "file open failed" << endl;
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
            }
    }
    double res1;
    vector<double> get_neighbors_attention_vec(int e, int rel)
    {
	vector<double> attention_vec(n);
	vector<double> weight(entity_neighbors[make_pair(e, rel)].size());
	double sum = 0;
    	for (int i = 0; i<entity_neighbors[make_pair(e, rel)].size(); i++)
	{
		sum += exp(get_sum(e, entity_neighbors[make_pair(e, rel)][i], rel));
		weight.push_back(exp(get_sum(e, entity_neighbors[make_pair(e, rel)][i], rel)));
	}
	for (int ii=0; ii<n; ii++)
	{
		attention_vec[ii] = 0.0;
		for (int i=0; i<entity_neighbors[make_pair(e, rel)].size(); i++)
			attention_vec[ii] += weight[i]/sum*entity_vec[entity_neighbors[make_pair(e, rel)][i]][ii];
	}
	return attention_vec;
    }
    double get_sum(int e1, int e2, int rel)
    {
	double sum=0;
	if (L1_flag)
		for (int ii=0; ii<n; ii++)
			sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	else
		for (int ii=0; ii<n; ii++)
			sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	return sum;
    }
	
    double calc_sum(int e1,int e2,int rel)
    {
	vector<double> neighbors_e1 = get_neighbors_attention_vec(e1, rel);
	vector<double> neighbors_e2 = get_neighbors_attention_vec(e2, rel);
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            	sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]) + fabs(neighbors_e2[ii]-neighbors_e1[ii]-relation_vec[rel][ii] + con_alpha*(neighbors_e2[ii]-entity_vec[e2][ii] + neighbors_e1[ii]-entity_vec[e1][ii]));
        else
        	for (int ii=0; ii<n; ii++)
            	sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]) + sqr(neighbors_e2[ii]-neighbors_e1[ii]-relation_vec[rel][ii] + con_alpha*(neighbors_e2[ii]-entity_vec[e2][ii] + neighbors_e1[ii]-entity_vec[e1][ii]));
        return sum;
    }
    void gradient(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
	vector<double> neighbors_e1_a = get_neighbors_attention_vec(e1_a, rel_a);
	vector<double> neighbors_e2_a = get_neighbors_attention_vec(e2_a, rel_a);
        vector<double> neighbors_e1_b = get_neighbors_attention_vec(e1_b, rel_b);
        vector<double> neighbors_e2_b = get_neighbors_attention_vec(e2_b, rel_b);
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]+neighbors_e2_a[ii]-neighbors_e1_a[ii]-relation_vec[rel_a][ii]+con_alpha*(neighbors_e2_a[ii]-entity_vec[e2_a][ii]+neighbors_e1_a[ii]-entity_vec[e1_a][ii]));
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]+neighbors_e2_b[ii]-neighbors_e1_b[ii]-relation_vec[rel_b][ii]+con_alpha*(neighbors_e2_b[ii]-entity_vec[e2_b][ii]+neighbors_e1_b[ii]-entity_vec[e2_a][ii]));
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;
        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
        	res+=margin+sum1-sum2;
        	gradient( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }
};

Train train;
void prepare()
{
    FILE* f1 = fopen((data_path+"entity2id.txt").c_str(),"r");
    FILE* f2 = fopen((data_path+"relation2id.txt").c_str(),"r");
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
	string st=buf;
	entity2id[st]=x;
	id2entity[x]=st;
	entity_num++;
    }
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
	string st=buf;
	relation2id[st]=x;
	id2relation[x]=st;
	relation_num++;
    }
    FILE* f_kb = fopen((data_path+"train.txt").c_str(),"r");
    int cnt = 0;
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
	    cnt++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        train.add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
	//cout << right_num[i] << " ";
    }
//    for (map<int, int>::iterator it = right_entity[0].begin(); it!=right_entity[0].end(); it++)
//	cout << it->first << ":" << it->second << ",";
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int method = 1;
    int n = 100;
    double rate = 0.001;
    double margin = 1;
    double con_alpha = 0.01;
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-con_alpha", argc, argv)) > 0) con_alpha = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    cout<<"con_alpha = "<<con_alpha<<endl;
    prepare();
    train.run(n,rate,margin,method, con_alpha);
}
