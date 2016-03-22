/* * * * *
 *  AzsLinear.cpp 
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

#include "AzsLinear.hpp"
#include "AzsSvrg.hpp"

/*--------------------------------------------------------*/
void AzsLinear::train_test(const char *inp_param)
{
  const char *param = inp_param; 
  AzBytArr s_tmp; 
  if (*param == '@') {
    /*---  read paramaters from a file  ---*/
    const char *fn = param+1; 
    AzParam::read(param+1, &s_tmp); 
    param = s_tmp.c_str(); 
    cout << param << endl; 
  }
  
  /*---  parse parameters  ---*/
  AzParam azp(param); 
  resetParam(azp); 
  printParam(log_out); 
  azp.check(log_out, &s_param); 

  /*---  read data  ---*/
  AzSvDataS trn_ds, tst_ds; 
  AzTimeLog::print("Reading training data ... ", log_out); 
  trn_ds.read(s_train_x_fn.c_str(), s_train_tar_fn.c_str()); 
  AzTimeLog::print("Reading test data ... ", log_out); 
  tst_ds.read(s_test_x_fn.c_str(), s_test_tar_fn.c_str()); 

  if (!do_no_intercept) { /* append a feature with a constant 1 */
    AzTimeLog::print("Adding a constant feature ...", log_out); 
    trn_ds.append_const(1); 
    tst_ds.append_const(1); 
  }
  AzDvect v_trn_y(trn_ds.targets()), v_tst_y(tst_ds.targets()); 
  AzBytArr s_num("#train="); s_num.cn(trn_ds.size()); s_num.c(", #test="); s_num.cn(tst_ds.size()); 
  
  /*---  sparse?  dense?  ---*/
  AzDSmat trn_dsm, tst_dsm; 
  AzDmat md_trn, md_tst; 
  if (is_dense_matrix(trn_ds.feat())) {
    /*---  treat it as dense data  ---*/
    AzTimeLog::print("Processing as DENSE data ... ", s_num.c_str(), log_out); 
    md_trn.set(trn_ds.feat()); 
    trn_ds.destroy(); 
    trn_dsm.reset(&md_trn); 
    md_tst.set(tst_ds.feat()); 
    tst_ds.destroy(); 
    tst_dsm.reset(&md_tst); 
  }
  else {
    /*---  treat it as sparse data  ---*/
    AzTimeLog::print("Processing as SPARSE data ... ", s_num.c_str(), log_out); 
    trn_dsm.reset(trn_ds.feat()); 
    tst_dsm.reset(tst_ds.feat()); 
  }

  if (do_regress) {
    AzsSvrg svrg; 
    svrg.train_test_regress(s_param.c_str(), &trn_dsm, &v_trn_y, &tst_dsm, &v_tst_y);  
  }
  else {
    /*---  check and set training labels  ---*/
    AzIntArr ia_trn_lab; 
    set_labels(&v_trn_y, ia_trn_lab); 
    int class_num = ia_trn_lab.max() + 1; 

    /*---  check test labels  ---*/
    AzIntArr ia_tst_lab; 
    set_labels(&v_tst_y, ia_tst_lab); 
    if (ia_tst_lab.max() >= class_num) {
      throw new AzException(AzInputError, "AzsLinear::train_test", "test label is out of range"); 
    }
    AzPrint::writeln(log_out, "#class: ", class_num);   
  
    AzsSvrg svrg; 
    svrg.train_test_classif(s_param.c_str(), &trn_dsm, &ia_trn_lab, &tst_dsm, &ia_tst_lab, class_num); 
  }
}
 
/*--------------------------------------------------------*/
bool AzsLinear::is_dense_matrix(const AzSmat *m) const
{
  if (do_dense) return true; 
  if (do_sparse) return false; 
  AzTimeLog::print("Checking data sparsity ... ", log_out); 
  double nz_ratio; 
  m->nonZeroNum(&nz_ratio); 
  AzBytArr s("nonzero ratio = "); s.cn(nz_ratio); 
  AzPrint::writeln(log_out, s); 
  return (nz_ratio >= 0.6666); 
}

/*--------------------------------------------------------*/
void AzsLinear::set_labels(const AzDvect *v_mc, AzIntArr &ia_lab) const 
{
  const char *eyec = "AzsLinear::set_labels"; 
  ia_lab.reset(v_mc->rowNum(), -1); 
  const double *mc = v_mc->point(); 
  int dx; 
  for (dx = 0; dx < v_mc->rowNum(); ++dx) {
    int lab = (int)mc[dx]; 
    if (lab != mc[dx]) throw new AzException(AzInputError, eyec, "class# must be integer"); 
    if (lab < 0) throw new AzException(AzInputError, eyec, "class# must be non-negative"); 
    ia_lab.update(dx, lab); 
  }
}

/*--------------------------------------------------------*/
#define kw_do_no_intercept "NoIntercept"
#define kw_train_x_fn "train_x_fn="
#define kw_train_tar_fn "train_target_fn="
#define kw_test_x_fn "test_x_fn="
#define kw_test_tar_fn "test_target_fn="
#define kw_do_dense "DenseData"
#define kw_do_sparse "SparseData"
#define kw_do_regress "Regression"
/*--------------------------------------------------------*/
void AzsLinear::resetParam(AzParam &azp)
{
  azp.swOn(&do_no_intercept, kw_do_no_intercept, false); 
  azp.vStr(kw_train_x_fn, &s_train_x_fn); 
  azp.vStr(kw_test_x_fn, &s_test_x_fn);
  azp.vStr(kw_train_tar_fn, &s_train_tar_fn);
  azp.vStr(kw_test_tar_fn, &s_test_tar_fn);
  azp.swOn(&do_dense, kw_do_dense); 
  azp.swOn(&do_sparse, kw_do_sparse);     
  azp.swOn(&do_regress, kw_do_regress); 
}

/*--------------------------------------------------------*/
void AzsLinear::printParam(const AzOut &out) const 
{
  if (out.isNull()) return; 

  AzPrint o(out); 
  o.printSw(kw_do_no_intercept, do_no_intercept); 
  o.printV(kw_train_x_fn, s_train_x_fn); 
  o.printV(kw_train_tar_fn, s_train_tar_fn); 
  o.printV(kw_test_x_fn, s_test_x_fn); 
  o.printV(kw_test_tar_fn, s_test_tar_fn);   
  o.printSw(kw_do_dense, do_dense); 
  o.printSw(kw_do_sparse, do_sparse); 
  o.printSw(kw_do_regress, do_regress);   
  o.ppEnd(); 
   
  throw_if_empty(kw_train_x_fn, s_train_x_fn); 
  throw_if_empty(kw_train_tar_fn, s_train_tar_fn); 
  throw_if_empty(kw_test_x_fn, s_test_x_fn); 
  throw_if_empty(kw_test_tar_fn, s_test_tar_fn);   
  if (do_dense && do_sparse) {
    AzBytArr s(kw_do_sparse); s.c(" and "); s.c(kw_do_dense); s.c(" are mutually exclusive."); 
    throw new AzException(AzInputError, "AzsLinear::resetParam", s.c_str()); 
  }
}

/*------------------------------------------------------------------*/
void AzsLinear::printHelp(const AzOut &out) const
{
  AzHelp h(out); 
  h.writeln_header("Usage:  linsvrg  parameters | @parameter_filename"); 
  AzBytArr s_kw("  parameters"); 
  AzBytArr s_desc("keyword-value pairs (e.g., \"num_iterations=50\") and options (e.g., \"ShowLoss\") delimited by \",\"."); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  s_kw.reset("  parameter_filename"); 
  s_desc.reset("If the argument begins with \'@\', the rest is regarded as the parameter filename; e.g., if the argument is \"@test.param\", \"test.param\" will be scanned for parameters."); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  h.nl();  
  h.writeln_header("In the parameter description below, \"*\" indicates the required parameters that cannot be omitted.");
  h.nl(); 
 
  h.item_required(kw_train_x_fn, "Training data feature file.  See README for the format."); 
  h.item_required(kw_train_tar_fn, "Training target file: class labels 0,1,... (classification) or target values (regression); one per line."); 
  h.item_required(kw_test_x_fn, "Test data feature file.  See README for the format."); 
  h.item_required(kw_test_tar_fn, "Test target file: class labels 0,1,... (classification) or target values (regression); one per line.");   
  h.item(kw_do_regress, "Specify this if the task is regression.", " classification task"); 
  h.item(kw_do_no_intercept, "Do not use intercept."); 
  h.item(kw_do_dense, "Use dense matrices for features.", 
         " automatically determined based on data sparseness"); 
  h.item(kw_do_sparse, "Use sparse matrices for features.", 
         " automatically determined based on data sparseness"); 
  h.nl(); 
  
  AzsSvrg::printHelp(h); 
  
  h.end();  
}
