#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
extern void dsyevr_(const char*,const char*,const char*,int*,double*,int*,double*,double*,int*,int*,double*,int*,double*,double*,int*,int*,double*,int*,int*,int*,int*,int,int,int);
extern void dsytrd_(const char*,int*,double*,int*,double*,double*,double*,double*,int*,int*,int);
extern void dsterf_(int*,double*,double*,int*);
extern double dlamch_(const char*);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
 int batches=argc>1?atoi(argv[1]):15;
 int ns[]={32,64,128,256,512};
 for(int k=0;k<5;k++){int n=ns[k];int reps=n<=64?30:n<=128?10:4;
  double*B=malloc(8*n*n),*A=malloc(8*n*n),*W=malloc(8*n*n),*Z=malloc(8*n*n),*w=malloc(8*n);
  uint64_t x=0x9E3779B97F4A7C15ULL;
  for(int i=0;i<n*n;i++){x^=x<<13;x^=x>>7;x^=x<<17;B[i]=(double)(x>>11)/9007199254740992.0-0.5;}
  for(int j=0;j<n;j++)for(int i=0;i<n;i++){double s=0;for(int l=0;l<n;l++)s+=B[i+l*n]*B[j+l*n];A[i+j*n]=s+(i==j?n:0);}
  int lw=-1,liw=-1,info,m,il=1,iu=n;double wq,vl=0,vu=0;int iwq;int*isuppz=malloc(8*n*2);
  double abst=-1;
  dsyevr_("V","A","L",&n,W,&n,&vl,&vu,&il,&iu,&abst,&m,w,Z,&n,isuppz,&wq,&lw,&iwq,&liw,&info,1,1,1);
  lw=(int)wq;liw=iwq;double*work=malloc(8*lw);int*iw=malloc(4*liw);
  double *tv=malloc(8*batches),*tvec=malloc(8*batches),*ttr=malloc(8*batches),*tst=malloc(8*batches);
  double*d=malloc(8*n),*e=malloc(8*n),*tau=malloc(8*n),*d0=malloc(8*n),*e0=malloc(8*n);
  int lwt=-1;dsytrd_("L",&n,W,&n,d,e,tau,&wq,&lwt,&info,1);lwt=(int)wq;double*wt=malloc(8*lwt);
  for(int b=0;b<batches;b++){
   double best=1e30;
   for(int r=0;r<reps;r++){memcpy(W,A,8*n*n);double t=now();dsyevr_("N","A","L",&n,W,&n,&vl,&vu,&il,&iu,&abst,&m,w,Z,&n,isuppz,work,&lw,iw,&liw,&info,1,1,1);double dt=now()-t;if(dt<best)best=dt;}
   tv[b]=best;best=1e30;
   for(int r=0;r<reps;r++){memcpy(W,A,8*n*n);double t=now();dsyevr_("V","A","L",&n,W,&n,&vl,&vu,&il,&iu,&abst,&m,w,Z,&n,isuppz,work,&lw,iw,&liw,&info,1,1,1);double dt=now()-t;if(dt<best)best=dt;}
   tvec[b]=best;best=1e30;
   for(int r=0;r<reps;r++){memcpy(W,A,8*n*n);double t=now();dsytrd_("L",&n,W,&n,d0,e0,tau,wt,&lwt,&info,1);double dt=now()-t;if(dt<best)best=dt;}
   ttr[b]=best;best=1e30;
   for(int r=0;r<reps;r++){memcpy(d,d0,8*n);memcpy(e,e0,8*n);double t=now();dsterf_(&n,d,e,&info);double dt=now()-t;if(dt<best)best=dt;}
   tst[b]=best;}
  qsort(tv,batches,8,cmp);qsort(tvec,batches,8,cmp);qsort(ttr,batches,8,cmp);qsort(tst,batches,8,cmp);
  printf("n=%d values_us=%.2f dsytrd_us=%.2f dsterf_us=%.2f vectors_us=%.2f\n",n,tv[batches/2],ttr[batches/2],tst[batches/2],tvec[batches/2]);}
}
