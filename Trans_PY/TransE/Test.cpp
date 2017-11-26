#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;
bool L1_flag=1;

string version;
string data_path = "../FB15k/";
string trainortest = "test";

map<string, string> fb2wiki;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;
int n= 100;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<string,double> a, pair<string,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]);
        return sum;
    }
    void run()
    {
        FILE* f1 = fopen(("relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen(("entity2vec."+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);

        //for (int testid = 0; testid<fb_l.size(); testid+=1)
	{
		int testid = 0;
                
		int l = fb_h[testid];
		//int rel = fb_r[testid];
		//double tmp = calc_sum(h,l,rel);
		//rel_num[rel]+=1;
		vector<pair<string,double> > a;
		for (int i=0; i<entity_num; i++)
		{
			double sum = calc_sum(i,l);
			a.push_back(make_pair(fb2wiki[id2entity[i]],sum));
		}
		sort(a.begin(),a.end(),cmp);
		cout << fb2wiki[id2entity[l]] << ": " << endl; 
		for(int i = 0; i < a.size() && i < 10; i++)
      				cout << a[i].first << " " << a[i].second << endl;
        }
    }

};
Test test;

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
		mid2type[st]="None";
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    FILE* f_kb = fopen((data_path+"test.txt").c_str(),"r");
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
        	cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen((data_path+"train.txt").c_str(),"r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
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
        }

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen((data_path+"valid.txt").c_str(),"r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
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
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
    
   FILE* f2w = fopen((data_path+"f2w.nt").c_str(), "r");
   if(f2w == NULL)
	cout << "open f2w.nt falied" << endl;
   while(fscanf(f2w,"%s", buf) == 1)
   {
	string fb = buf;
        fscanf(f2w, "%s", buf);
        string wiki = buf;
        fb2wiki[fb] = wiki; 
   }
   fclose(f2w); 
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        prepare();
        test.run();
    }
}

