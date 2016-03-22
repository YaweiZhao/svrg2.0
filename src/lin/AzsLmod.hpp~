/* * * * *
 *  AzsLmod.hpp 
 *  Copyright (C) 2013 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#ifndef _AZS_LMOD_HPP_
#define _AZS_LMOD_HPP_

#include "AzUtil.hpp"
#include "AzDmat.hpp"

/*---  loss type  ---*/
enum AzsLossType {
  AzsLoss_Square = 0,    /* 1/2 (p-y)^2 */
  AzsLoss_Logistic = 1,
  AzsLoss_None = 2, 
};

/*---  linear model  ---*/
class AzsLmod {
protected:
  AzDmat m_w; /* weights */
  double ws;  /* weight scale; used only when momentum is not used */

public:
  AzsLmod() : ws(1) {}
  /*this function is used for regression*/  double* getWeights(int dim)
  {
    double * weights = new double (dim);
    for(int i=0;i<dim;i++)
    {
      weights[i] = m_w.col(0)->get(i);
    }
    return weights;
  }

  virtual void reset_weights(int dim, int class_num); 
  virtual void reset(const AzDmat *inp_m_w, double inp_ws) {
    m_w.set(inp_m_w); 
    ws = inp_ws; 
  }
 
  inline void apply(const AzDSmat *m_x, int col, AzDvect *v_pred) const {  
    if (m_x->is_sparse()) apply(m_x->sparse()->col(col), v_pred); 
    else                  apply(m_x->dense()->col(col), v_pred); 
  }
  /* <class V>: AzSvect (sparse vector) | AzDvect (dense vector) */
  template <class V> void apply(const V *v_x, AzDvect *v_pred) const {
    v_pred->reform(classNum()); 
    double *pred = v_pred->point_u(); 
    int cx; 
    for (cx = 0; cx < classNum(); ++cx) {
      pred[cx] = ws*m_w.col(cx)->innerProduct(v_x); 
    }
  }
  template <class V> double apply(const V *v_x) const {
    if (classNum() != 1) throw new AzException("AzsLmod::apply", "apply(x) is only for binary classification."); 
    return ws*m_w.col(0)->innerProduct(v_x); 
  }

  /*------------------------------------------------------------*/     
  inline void apply(const AzDSmat *dsm, AzDmat *m_pred) const {
    if (dsm->is_sparse()) apply(dsm->sparse(), m_pred); 
    else                  apply(dsm->dense(), m_pred); 
  }
  /* <class M>: AzSmat (sparse matrix)  | AzDmat (dense matrix)  */
  template <class M> void apply(const M *m_x, AzDmat *m_pred) const {
    int data_size = m_x->colNum(); 
    m_pred->reform(classNum(), data_size); 
    int dx; 
    for (dx = 0; dx < data_size; ++dx) apply(m_x->col(dx), m_pred->col_u(dx));
  }

  /*------------------------------------------------------------*/    
  inline double test_classif(const AzDSmat *dsm, const AzIntArr *ia_lab) const {
    if      (dsm->is_sparse()) return test_classif(dsm->sparse(), ia_lab); 
    else if (dsm->is_dense()) return test_classif(dsm->dense(), ia_lab); 
    return 0; 
  }  
  template <class M> double test_classif(const M *m_x, const AzIntArr *ia_lab) const {
    if (classNum() == 1) return test_classif_bin<M>(m_x, ia_lab); 
    else                 return test_classif_multi<M>(m_x, ia_lab); 
  } 
  
  /*------------------------------------------------------------*/   
  inline int classNum() const { return m_w.colNum(); }   
  
  /*------------------------------------------------------------*/      
  inline void reset_ws() { ws = 1; }   
  inline void check_ws(const char *msg) const {
    if (ws != 1) throw new AzException("AzsLmod::check_ws", "weight scale must have been flushed before calling:", msg); 
  }
  inline void flush_ws() {
    if (ws != 1) {
      m_w.multiply(ws); 
      ws = 1; 
    }
  }

  /*------------------------------------------------------------*/     
  inline void get_deriv_classif(AzsLossType loss_type, const AzDSmat *dsm, int col, int cls, AzDvect *v_deriv) const {
    if      (dsm->is_sparse()) get_deriv_classif(loss_type, dsm->sparse()->col(col), cls, v_deriv); 
    else                       get_deriv_classif(loss_type, dsm->dense()->col(col), cls, v_deriv); 
  }
  template <class V> void get_deriv_classif(AzsLossType loss_type, 
                                    const V *v_x, /* data point */
                                    int cls, /* true class */
                                    AzDvect *v_deriv) const { /* output: derivatives */
    AzDvect v_pred; 
    apply(v_x, &v_pred); 
    _get_deriv_classif(loss_type, &v_pred, cls, v_deriv);     
  }                       
 
  /*------------------------------------------------------------*/     
  inline void get_deriv_regress(AzsLossType loss_type, const AzDSmat *dsm, int col, double y, AzDvect *v_deriv) const {
    if      (dsm->is_sparse()) get_deriv_regress(loss_type, dsm->sparse()->col(col), y, v_deriv); 
    else                       get_deriv_regress(loss_type, dsm->dense()->col(col), y, v_deriv); 
  }
  template <class V> void get_deriv_regress(AzsLossType loss_type, 
                                    const V *v_x, /* data point */
                                    double y, /* true target */
                                    AzDvect *v_deriv) const { /* output: derivative */
    double p = apply(v_x); 
    double deriv = getLossd(loss_type, p, y); 
    v_deriv->reform(1); v_deriv->set(0, deriv);   
  } 
  
  /*------------------------------------------------------------*/  
  static AzsLossType lossType(const char *param); 
  static const char *lossName(AzsLossType loss_type); 
  static void allLoss_str(AzBytArr &s); 
  static double getLossd(AzsLossType loss_type, double p, double y); 
  static double getLoss(AzsLossType loss_type, double p, double y); 
  static double getLoss_classif(AzsLossType loss_type, const AzDvect *v_pred, int cls); 
  
  /*--------------------------------------------------------*/
  inline double getLossAvg_classif(AzsLossType loss_type, const AzDSmat *m_x, const AzIntArr *ia_lab) const {
    if (m_x->is_sparse()) return getLossAvg_classif(loss_type, m_x->sparse(), ia_lab); 
    else                  return getLossAvg_classif(loss_type, m_x->dense(), ia_lab); 
  }
  template <class M>
  double getLossAvg_classif(AzsLossType loss_type, const M *m_x, const AzIntArr *ia_lab) const {
    double loss = 0; 
    int dx;
    for (dx = 0; dx < m_x->colNum(); ++dx) {
      AzDvect v_pred; 
      apply(m_x->col(dx), &v_pred); 
      int cls = ia_lab->get(dx);  
      loss += getLoss_classif(loss_type, &v_pred, cls);  
    }  
    loss /= (double)m_x->colNum(); 
    return loss; 
  }
  
  /*--------------------------------------------------------*/
  inline double getLossAvg_regress(AzsLossType loss_type, const AzDSmat *m_x, const AzDvect *v_y) const {
    if (m_x->is_sparse()) return getLossAvg_regress(loss_type, m_x->sparse(), v_y); 
    else                  return getLossAvg_regress(loss_type, m_x->dense(), v_y); 
  }  
  template <class M>
  double getLossAvg_regress(AzsLossType loss_type, const M *m_x, const AzDvect *v_y) const {
    double loss = 0; 
    int dx;
    for (dx = 0; dx < m_x->colNum(); ++dx) {
      double p = apply(m_x->col(dx)); 
      double y = v_y->get(dx); 
      loss += getLoss(loss_type, p, y); 
    } 

    loss /= (double)m_x->colNum(); 
    return loss; 
  }
  
  /*--------------------------------------------------------*/
  void write_pred(const AzDSmat *m_x, const char *out_fn) const {
    AzDmat m_pred; 
    apply(m_x, &m_pred); 
    AzFile file(out_fn); 
    int digits = 7; 
    m_pred.writeText(out_fn, digits); 
  }
  
protected:   
  static void get_prob(const AzDvect *v_pred, AzDvect *v_prob); 

  /*--------------------------------------------------------*/
  template <class M> double test_classif_bin(const M *m_x, const AzIntArr *ia_lab) const {
    int correct = 0; 
    int dx; 
    for (dx = 0; dx < m_x->colNum(); ++dx) {
      double p = apply(m_x->col(dx));  
      int cls = (p >= 0) ? 0 : 1;    
      if (cls == ia_lab->get(dx)) ++correct; 
    }
    double acc = correct / (double)m_x->colNum(); 
    return acc; 
  }

  /*--------------------------------------------------------*/
  template <class M> double test_classif_multi(const M *m_x, const AzIntArr *ia_lab) const {
    int correct = 0; 
    int dx; 
    for (dx = 0; dx < m_x->colNum(); ++dx) {
      AzDvect v_pred; 
      apply(m_x->col(dx), &v_pred); 
      int cls; 
      v_pred.max(&cls); 
      if (cls == ia_lab->get(dx)) ++correct; 
    }
    double acc = correct / (double)m_x->colNum(); 
    return acc; 
  }  
  
  static void _get_deriv_classif(AzsLossType loss_type, const AzDvect *v_pred, int cls, AzDvect *v_deriv);
}; 
#endif 
