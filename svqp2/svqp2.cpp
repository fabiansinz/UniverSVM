// -*- C++ -*-

/***********************************************************************
 * 
 *  LUSH Lisp Universal Shell
 *    Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *  Includes parts of TL3:
 *    Copyright (C) 1987-1999 Leon Bottou and Neuristique.
 *  Includes selected parts of SN3.2:
 *    Copyright (C) 1991-2001 AT&T Corp.
 * 
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
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

/***********************************************************************
 * $Id: svqp2.cpp,v 1.6 2005/11/16 03:58:41 agbs Exp $
 **********************************************************************/

//////////////////////////////////////
///
/// SVQP2 uses a gauss-seidel/smo method.
/// It also implements proper caching etc.
///
/// Leon Bottou - 2004
///
/////////////////////////////////////


#include "svqp2.h"

#include <math.h>
#include <stdio.h>
#include <stdarg.h>





// --------- utilities

template <class T>
class vec
{
private:
  T *d;
public:
  vec<T>(int n) : d(new T[n]) { }
  ~vec<T>() { delete [] d; }
  T& operator[](int n) { return d[n]; }
  const T& operator[](int n) const { return d[n]; }
  operator T*() { return d; }
  operator const T*() { return d; }
private:
  vec<T>(const vec<T>&);
  T& operator=(const vec<T>&);
};

template <class T> 
inline T min(T a, T b)
{
  if (a < b) 
    return a;
  else
    return b;
}

template <class T> 
inline T max(T a, T b)
{
  if (a < b) 
    return b;
  else
    return a;
}

template <class T>
inline void exch(T &v1, T &v2)
{
  T tmp = v1;
  v1 = v2;
  v2 = tmp;
}



// --------- messages

void
SVQP2::info(const char *s1, const char *s2, ...)
{
  if (verbosity > 0)
    {
      const char *fmt = s1;
      if (s2 && (verbosity>1 || vdots>=40 || !fmt))
	fmt = s2;
      if (fmt && fmt[0])
	{
	  va_list ap;
	  va_start(ap, s2);
	  vfprintf(stdout, fmt, ap);
	  va_end(ap);
	  fflush(stdout);
	  vdots += 1;
	  if (fmt[1])
	    vdots = 0;
	}
    }
}

int 
SVQP2::error(const char *message)
{
  if (verbosity>0 && vdots>0)
    fprintf(stdout,"\n");
  if (verbosity>0)
    fprintf(stdout,"SVQP2 error: %s\n", message);
  errmsg = message;
  return -1;
}



// --------- cache

struct SVQP2::Arow
{
  Arow  *prev;
  Arow  *next;
  int    ai;
  int    sz;
  float *d;
public:
  Arow(int);
  ~Arow();
  void refile(Arow *after);
  long resize(int nsz);
};

inline SVQP2::Arow::Arow(int ai)
  : prev(0), next(0), ai(ai), sz(0), d(0)
{
}

SVQP2::Arow::~Arow()
{
  delete [] d;
}

void 
SVQP2::Arow::refile(Arow *after)
{
  if (prev && next)
    {
      prev->next = next;
      next->prev = prev;
    }
  if (sz > 0)
    {
      next = after->next;
      prev = after;
      next->prev = this;
      prev->next = this;
    }
  else
    {
      prev = 0;
      next = 0;
    }
}

long
SVQP2::Arow::resize(int nsz)
{
  if (nsz != sz)
    {
      float *nd = 0;
      if (nsz > 0)
	{
	  nd = new float[nsz];
	  for (int i=0; i<nsz && i<sz; i++)
	    nd[i] = d[i];
	}
      long delta = (nsz - sz) * sizeof(float);
      sz = nsz;
      delete [] d;
      d = nd;
      return delta;
    }
  return 0;
}

void
SVQP2::cache_init()
{

  cache_fini();

  curcachesize = 0;
  hitcache = false;
  lru = new Arow(0);
  lru->next = lru;
  lru->prev = lru;
  vec<Arow*> c(n);
  for (int i=0; i<n; i++)
    c[i] = 0;

  for (int i=0; i<n; i++)
    {

      int ai = Aperm[i];
      int h = ai % n;

      Arow *p = c[h];
      while (p && p->ai != ai)
	p = p->next;
      if (! p)
	{
	  p = new Arow(ai);
	  p->next = c[h];
	  c[h] = p;
	}
      rows[i] = p;
    }

}

void
SVQP2::cache_fini()
{

  if (lru)
    {
      Arow *l = 0;
      for (int i=0; i<n; i++)
	rows[i]->prev = rows[i]->next = 0;
      for (int i=0; i<n; i++)
	if (rows[i] && !rows[i]->prev)
	  {
	    rows[i]->next = l;
	    l = rows[i];
	    l->prev = l;
	    rows[i] = 0;
	  }
      while (l)
	{
	  Arow *n = l;
	  l = l->next;
	  delete n;
	}
      curcachesize = 0;
      delete lru;
      lru = 0;
    }
}

long
SVQP2::cache_clean()
{
  if (curcachesize > maxcachesize)
    {
      if (! hitcache)
	info("@","@ Reached maxcachesize (%ld MB)\n",(maxcachesize>>20));
      hitcache = true;
      while ((curcachesize > maxcachesize) && (lru->prev != lru))
	{
	  Arow *p = lru->prev;
	  curcachesize += p->resize(0);
	  p->refile(0);
	}
    }
  return curcachesize;
}

void
SVQP2::cache_swap(int i, int j)
{
  if (lru)
    {
      exch(rows[i],rows[j]);
      Arow *row = lru->next;
      while (row != lru)
	{
	  Arow *nextrow = row->next;
	  if (i < row->sz)
	    {
	      if (j < row->sz)
		exch(row->d[i], row->d[j]);
	      else
		curcachesize += row->resize(i);
	    }
	  else 
	    {
	      if (j < row->sz)
		curcachesize += row->resize(j);
	    }
	  row = nextrow;
	}
    }
}

float *
SVQP2::getrow(int i, int len, bool hot)
{
  Arow *row = rows[i];
  int osz = row->sz;
  if (osz >= len)
    {
      if (hot)
	row->refile(lru);
      return row->d;
    }
  curcachesize += row->resize(len);
  float *d = row->d;
  int ai = Aperm[i];
  for (int j=osz; j<len; j++)
    d[j] = (*Afunction)(ai, Aperm[j], Aclosure);
  if (hot)
    row->refile(lru);
  else if (! osz)
    row->refile(lru->prev);
  return d;
}

void 
SVQP2::swap(int i, int j)
{
  if (i != j)
    {
      exch(Aperm[i], Aperm[j]);
      exch(b[i], b[j]);
      exch(cmin[i], cmin[j]);
      exch(cmax[i], cmax[j]);
      exch(x[i], x[j]);
      exch(g[i], g[j]);
      exch(pivot[i], pivot[j]);
      cache_swap(i,j);
    }
}

void
SVQP2::unswap(void)
{
  vec<int> rpivot(n);
  for (int i=0; i<n; i++)
    rpivot[pivot[i]] = i;
  for (int i=0; i<n; i++)
    {
      int r = rpivot[i];
      swap(i, r);
      rpivot[pivot[i]] = i;
      rpivot[pivot[r]] = r;
    }
}



// --------- construct/destruct

SVQP2::SVQP2(int n)
  : n(n)
{
  dmem = new double[n * 5];
  imem = new int[n * 2];
  // setup
  Afunction = 0;
  Aclosure = 0;
  Aperm = imem;
  b = dmem;
  cmin = dmem + n;
  cmax = dmem + 2*n;
  x = dmem + 3*n;
  sumflag = true;
  verbosity = 1;
  epskt = 1e-15;
  epsgr = 1e-5;
  maxst = 1e+20;
  errmsg = 0;
  iter = 0;
  l = 0;
  g = dmem + 4*n;
  pivot = imem + n;
  curcachesize = 0;
  maxcachesize = 256*1024*1024;
  hitcache = false;
  vdots = 0;
  // init cache
  lru = 0;
  rows = new Arow* [n];
  // init vectors
  for (int i=0; i<n; i++)
    rows[i] = 0;
  for (int i=0; i<n; i++)
    Aperm[i] = pivot[i] = i;
  for (int i=0; i<n; i++)
    x[i] = cmin[i] = cmax[i] = b[i] = 0;
}


SVQP2::~SVQP2()
{
  cache_fini();
  delete [] dmem;
  delete [] imem;
  delete [] rows;
}


// --------- shrink

void
SVQP2::shrink()
{
  gmax = ((sumflag) ? -maxst : 0);
  gmin = ((sumflag) ?  maxst : 0);
  for (int i=0; i<l; i++)
    {
      double gi = g[i];
      if ((gi > gmax) && (x[i] < cmax[i]))
	gmax = gi;
      if ((gi < gmin) && (x[i] > cmin[i]))
	gmin = gi;
    }
  if (sumflag)
    {
      for (int i=0; i<l; i++)
        {
          double gi = g[i];
          if (((gi >= gmax) && (x[i] >= cmax[i]-epskt)) ||
              ((gi <= gmin) && (x[i] <= cmin[i]+epskt)) )
            swap(i--, --l);
        }
    }
  else
    {
      for (int i=0; i<l; i++)
        {
          double gi = g[i];
          if (((gi >= 0) && (x[i] >= cmax[i]-epskt)) ||
              ((gi <= 0) && (x[i] <= cmin[i]+epskt)) )
            swap(i--, --l);
        }
    }
}


void
SVQP2::unshrink(int s)
{
  s = min(s,n);
  if (l < s)
    {
      // recompute gradients
      for (int i=l; i<s; i++)
	g[i] = b[i];
      for (int j=0; j<s; j++)
	{
	  double xj = x[j];
	  if (xj != 0)
	    {
	      cache_clean();
	      float *arow = getrow(j, s, false);
	      for (int i=l; i<s; i++)
		g[i] -= xj * arow[i];
	    }
	}
    }
  // compute bounds
  l = s;
  double sum = 0;
  gmax = ((sumflag) ? -maxst : 0);
  gmin = ((sumflag) ?  maxst : 0);
  for (int i=0; i<l; i++)
    {
      double gi = g[i];
      double xi = x[i];
      if ((gi > gmax) && (x[i] < cmax[i]))
	gmax = gi;
      if ((gi < gmin) && (x[i] > cmin[i]))
	gmin = gi;
      sum += xi * (b[i] + g[i]);
    }
  w = sum / 2;
}



// --------- iterate

#define RESULT_FIN    0
#define RESULT_HIT    1
#define RESULT_SHRINK 2


int
SVQP2::iterate_gs1()
{
  int icount = 0;
  int pcount = 0;
  vec<bool> mark(l);
  for(int i=0; i<l; i++)
    mark[i] = false;
  // Iterate
  for(;;)
    {
      // Determine extreme gradients
      int imax = -1;
      int imin = -1;
      gmin = gmax = 0;
      for (int i=0; i<l; i++)
	{
	  double gi = g[i];
	  if ((gi > gmax) && (x[i] < cmax[i]))
	    {
	      gmax = gi;
	      imax = i;
	    }
	  if ((gi < gmin) && (x[i] > cmin[i]))
	    {
	      gmin = gi;
	      imin = i;
	    }
	}
      if (gmin + gmax < 0)
	imax = imin;
      bool down = (g[imax]<0);
      // Exit tests
      if (gmax - gmin < epsgr)
	return RESULT_FIN;
      icount += 1;
      if (! mark[imax]) 
	pcount += 2;
      mark[imax] = true;
      if (pcount<l && pcount+200<icount)
	return RESULT_SHRINK;
      // compute kernels
      cache_clean();
      float *rmax = getrow(imax, l);
      double step, ostep, curvature;
      bool   hit = true;
      if (down)
        step = x[imax] - cmin[imax];
      else
	step = cmax[imax] - x[imax];
      // compute newton step
      curvature = rmax[imax];
      if (curvature >= epskt)
        {
          ostep = fabs(g[imax]) / curvature;
	  if (ostep < step)
	    {
	      hit = false;
	      step = ostep;
	    }
        }
      else if (curvature + epskt < 0)
        return error("Function is not convex (negative curvature)");
      // update x and g
      if (down)
	step = - step;
      x[imax] += step;
      for (int j=0; j<l; j++)
	g[j] -= step * rmax[j];
      // stop on hit
      iter += 1;
      if (hit)
	return RESULT_HIT;
    }
}

int
SVQP2::iterate_gs2()
{
  int icount = 0;
  int pcount = 0;
  vec<bool> mark(l);
  for(int i=0; i<l; i++)
    mark[i] = false;
  // Iterate
  for(;;)
    {
      // Determine extreme gradients
      int imax = -1;
      int imin = -1;
      gmax = -maxst;
      gmin = maxst;
      for (int i=0; i<l; i++)
	{
	  double gi = g[i];
	  if ((gi > gmax) && (x[i] < cmax[i]))
	    {
	      gmax = gi;
	      imax = i;
	    }
	  if ((gi < gmin) && (x[i] > cmin[i]))
	    {
	      gmin = gi;
	      imin = i;
	    }
	}
      // Exit tests
      if (gmax - gmin < epsgr)
	return RESULT_FIN;
      icount += 1;
      if (! mark[imax]) 
	pcount += 1;
      mark[imax] = true;
      if (! mark[imin]) 
	pcount += 1;
      mark[imin] = true;
      if (pcount<l && pcount+200<icount)
	return RESULT_SHRINK;
      // compute kernels
      cache_clean();
      float *rmax = getrow(imax, l);
      float *rmin = getrow(imin, l);
      double step, ostep, curvature;
      bool   hit = true;
      // compute max step
      step = cmax[imax] - x[imax];
      ostep = x[imin] - cmin[imin];
      step = min(ostep,step);
      // newton step
      curvature = rmax[imax]+rmin[imin]-rmax[imin]-rmin[imax];
      if (curvature >= epskt)
        {
          ostep = (gmax - gmin) / curvature;
	  if (ostep < step)
	    {
	      hit = false;
	      step = ostep;
	    }
	}
      else if (curvature + epskt < 0)
	return error("Function is not convex (negative curvature)");
      // update x and g
      x[imax] += step;
      x[imin] -= step;
      for (int j=0; j<l; j++)
	g[j] -= step * ( rmax[j] - rmin[j] );
      // stop on hit
      iter += 1;
      if (hit)
	return RESULT_HIT;
    }
}



// --------- run

      
int
SVQP2::run(bool first, bool last)
{
  int status = 0;
  double gn = maxst;

  

  // check constraints
  for (int i = 0; i != n; ++i)
    if (x[i] - cmin[i] < -epskt || x[i] - cmax[i] > epskt){
      //      fprintf(stderr,"%g < %g or %g > %g or %g: \n",x[i],cmin[i],x[i],cmax[i],x[i]-cmin[i],x[i]-cmax[i]);
      return error("Initial x[] does not satisfy inequality constraint");
    }
  
  // pseudo optimization: I just unswaps the cache and does a cache_fini
  if (!first && last){
      printf("\nCleaning and unswapping  cache...\n");
      cache_fini();
      unswap();
      return 0;
  }

  // prepare in the first iteration
  if (first){
      printf("\nSetting up cache ...\n");
      cache_init();
  }


  vdots = 0;
  iter = 0;
  l = 0;
  // loop
  for(;;)
    {
      // incorporate
      unshrink(n);
      gn = max(0.0, gmax-gmin);
      info("*","* it:%d l:%d |g|:%f w:%f\n", iter, l, gn, w);
      // test termination
      if (gn < epsgr)
	break;
      // minimize
      for(;;)
	{
	  if (sumflag)
	    status = iterate_gs2();
	  else
	    status = iterate_gs1();
	  if (status <= RESULT_FIN)
	    break;
	  // shrink
	  int p = l;
	  shrink();
	  gn = max(0.0, gmax-gmin);
	  if (status == RESULT_SHRINK)
	    info(":",": it:%d l:%d |g|:%f\n", iter, l, gn);
	  else if (l < p)
	    info(".",". it:%d l:%d |g|:%f\n", iter, l, gn);
	  else if (verbosity >= 3)
	    info("!","! it:%d l:%d |g|:%f\n", iter, l, gn);
	}
      if (status < 0)
	break;
    }

  // finish if there's no iterated optimization
  if (first && last){
      printf("\nCleaning and unswapping  cache...\n");
      cache_fini();
      unswap();
  }
 
  if (status < 0)
    return status;
  if (vdots>0)
    info(0," it:%d l:%d |g|:%f w:%f\n", iter, l, gn, w);
  return iter;
}
