"""
dataprep.clean
==============

Functions for cleaning and validating the columns of a DataFrame.

The functions are imported lazily: ``import dataprep.clean`` is cheap, and a
function (together with the libraries it needs) is only loaded the first time
it is accessed, e.g. by ``from dataprep.clean import clean_email``.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # explicit imports for IDEs and static type checkers
    from .clean_ad_nrt import clean_ad_nrt, validate_ad_nrt
    from .clean_address import clean_address, validate_address
    from .clean_al_nipt import clean_al_nipt, validate_al_nipt
    from .clean_ar_cbu import clean_ar_cbu, validate_ar_cbu
    from .clean_ar_cuit import clean_ar_cuit, validate_ar_cuit
    from .clean_ar_dni import clean_ar_dni, validate_ar_dni
    from .clean_at_businessid import clean_at_businessid, validate_at_businessid
    from .clean_at_postleitzahl import clean_at_postleitzahl, validate_at_postleitzahl
    from .clean_at_tin import clean_at_tin, validate_at_tin
    from .clean_at_uid import clean_at_uid, validate_at_uid
    from .clean_at_vnr import clean_at_vnr, validate_at_vnr
    from .clean_au_abn import clean_au_abn, validate_au_abn
    from .clean_au_acn import clean_au_acn, validate_au_acn
    from .clean_au_tfn import clean_au_tfn, validate_au_tfn
    from .clean_az_voen import clean_az_voen, validate_az_voen
    from .clean_be_bis import clean_be_bis, validate_be_bis
    from .clean_be_eid import clean_be_eid, validate_be_eid
    from .clean_be_iban import clean_be_iban, validate_be_iban
    from .clean_be_nn import clean_be_nn, validate_be_nn
    from .clean_be_ogm_vcs import clean_be_ogm_vcs, validate_be_ogm_vcs
    from .clean_be_ssn import clean_be_ssn, validate_be_ssn
    from .clean_be_vat import clean_be_vat, validate_be_vat
    from .clean_bg_egn import clean_bg_egn, validate_bg_egn
    from .clean_bg_pnf import clean_bg_pnf, validate_bg_pnf
    from .clean_bg_vat import clean_bg_vat, validate_bg_vat
    from .clean_bic import clean_bic, validate_bic
    from .clean_bitcoin import clean_bitcoin, validate_bitcoin
    from .clean_br_cnpj import clean_br_cnpj, validate_br_cnpj
    from .clean_br_cpf import clean_br_cpf, validate_br_cpf
    from .clean_by_unp import clean_by_unp, validate_by_unp
    from .clean_ca_bc_phn import clean_ca_bc_phn, validate_ca_bc_phn
    from .clean_ca_bn import clean_ca_bn, validate_ca_bn
    from .clean_ca_sin import clean_ca_sin, validate_ca_sin
    from .clean_casrn import clean_casrn, validate_casrn
    from .clean_cfi import clean_cfi, validate_cfi
    from .clean_ch_esr import clean_ch_esr, validate_ch_esr
    from .clean_ch_ssn import clean_ch_ssn, validate_ch_ssn
    from .clean_ch_uid import clean_ch_uid, validate_ch_uid
    from .clean_ch_vat import clean_ch_vat, validate_ch_vat
    from .clean_cl_rut import clean_cl_rut, validate_cl_rut
    from .clean_cn_ric import clean_cn_ric, validate_cn_ric
    from .clean_cn_uscc import clean_cn_uscc, validate_cn_uscc
    from .clean_co_nit import clean_co_nit, validate_co_nit
    from .clean_country import clean_country, validate_country
    from .clean_cr_cpf import clean_cr_cpf, validate_cr_cpf
    from .clean_cr_cpj import clean_cr_cpj, validate_cr_cpj
    from .clean_cr_cr import clean_cr_cr, validate_cr_cr
    from .clean_cu_ni import clean_cu_ni, validate_cu_ni
    from .clean_currency import clean_currency, validate_currency
    from .clean_cusip import clean_cusip, validate_cusip
    from .clean_cy_vat import clean_cy_vat, validate_cy_vat
    from .clean_cz_bankaccount import clean_cz_bankaccount, validate_cz_bankaccount
    from .clean_cz_dic import clean_cz_dic, validate_cz_dic
    from .clean_cz_rc import clean_cz_rc, validate_cz_rc
    from .clean_date import clean_date, validate_date
    from .clean_de_handelsregisternummer import (
        clean_de_handelsregisternummer,
        validate_de_handelsregisternummer,
    )
    from .clean_de_idnr import clean_de_idnr, validate_de_idnr
    from .clean_de_leitweg import clean_de_leitweg, validate_de_leitweg
    from .clean_de_stnr import clean_de_stnr, validate_de_stnr
    from .clean_de_vat import clean_de_vat, validate_de_vat
    from .clean_de_wkn import clean_de_wkn, validate_de_wkn
    from .clean_df import clean_df
    from .clean_df_gui import clean_df_gui
    from .clean_dk_cpr import clean_dk_cpr, validate_dk_cpr
    from .clean_dk_cvr import clean_dk_cvr, validate_dk_cvr
    from .clean_do_cedula import clean_do_cedula, validate_do_cedula
    from .clean_do_ncf import clean_do_ncf, validate_do_ncf
    from .clean_do_rnc import clean_do_rnc, validate_do_rnc
    from .clean_duplication import clean_duplication
    from .clean_dz_nif import clean_dz_nif, validate_dz_nif
    from .clean_ean import clean_ean, validate_ean
    from .clean_ec_ci import clean_ec_ci, validate_ec_ci
    from .clean_ec_ruc import clean_ec_ruc, validate_ec_ruc
    from .clean_ee_ik import clean_ee_ik, validate_ee_ik
    from .clean_ee_kmkr import clean_ee_kmkr, validate_ee_kmkr
    from .clean_ee_registrikood import clean_ee_registrikood, validate_ee_registrikood
    from .clean_eg_tn import clean_eg_tn, validate_eg_tn
    from .clean_email import clean_email, validate_email
    from .clean_es_cae import clean_es_cae, validate_es_cae
    from .clean_es_ccc import clean_es_ccc, validate_es_ccc
    from .clean_es_cif import clean_es_cif, validate_es_cif
    from .clean_es_cups import clean_es_cups, validate_es_cups
    from .clean_es_dni import clean_es_dni, validate_es_dni
    from .clean_es_iban import clean_es_iban, validate_es_iban
    from .clean_es_nie import clean_es_nie, validate_es_nie
    from .clean_es_nif import clean_es_nif, validate_es_nif
    from .clean_es_postal_code import clean_es_postal_code, validate_es_postal_code
    from .clean_es_referenciacatastral import (
        clean_es_referenciacatastral,
        validate_es_referenciacatastral,
    )
    from .clean_eu_at_02 import clean_eu_at_02, validate_eu_at_02
    from .clean_eu_banknote import clean_eu_banknote, validate_eu_banknote
    from .clean_eu_ecnumber import clean_eu_ecnumber, validate_eu_ecnumber
    from .clean_eu_eic import clean_eu_eic, validate_eu_eic
    from .clean_eu_excise import clean_eu_excise, validate_eu_excise
    from .clean_eu_nace import clean_eu_nace, validate_eu_nace
    from .clean_eu_oss import clean_eu_oss, validate_eu_oss
    from .clean_eu_vat import clean_eu_vat, validate_eu_vat
    from .clean_fi_alv import clean_fi_alv, validate_fi_alv
    from .clean_fi_associationid import clean_fi_associationid, validate_fi_associationid
    from .clean_fi_hetu import clean_fi_hetu, validate_fi_hetu
    from .clean_fi_veronumero import clean_fi_veronumero, validate_fi_veronumero
    from .clean_fi_ytunnus import clean_fi_ytunnus, validate_fi_ytunnus
    from .clean_figi import clean_figi, validate_figi
    from .clean_fo_vn import clean_fo_vn, validate_fo_vn
    from .clean_fr_accise import clean_fr_accise, validate_fr_accise
    from .clean_fr_nif import clean_fr_nif, validate_fr_nif
    from .clean_fr_nir import clean_fr_nir, validate_fr_nir
    from .clean_fr_rcs import clean_fr_rcs, validate_fr_rcs
    from .clean_fr_siren import clean_fr_siren, validate_fr_siren
    from .clean_fr_siret import clean_fr_siret, validate_fr_siret
    from .clean_fr_tva import clean_fr_tva, validate_fr_tva
    from .clean_gb_nhs import clean_gb_nhs, validate_gb_nhs
    from .clean_gb_sedol import clean_gb_sedol, validate_gb_sedol
    from .clean_gb_upn import clean_gb_upn, validate_gb_upn
    from .clean_gb_utr import clean_gb_utr, validate_gb_utr
    from .clean_gb_vat import clean_gb_vat, validate_gb_vat
    from .clean_gh_tin import clean_gh_tin, validate_gh_tin
    from .clean_gn_nifp import clean_gn_nifp, validate_gn_nifp
    from .clean_gr_amka import clean_gr_amka, validate_gr_amka
    from .clean_gr_vat import clean_gr_vat, validate_gr_vat
    from .clean_grid import clean_grid, validate_grid
    from .clean_gs1_128 import clean_gs1_128, validate_gs1_128
    from .clean_gt_nit import clean_gt_nit, validate_gt_nit
    from .clean_headers import clean_headers
    from .clean_hr_oib import clean_hr_oib, validate_hr_oib
    from .clean_hu_anum import clean_hu_anum, validate_hu_anum
    from .clean_iban import clean_iban, validate_iban
    from .clean_id_nik import clean_id_nik, validate_id_nik
    from .clean_id_npwp import clean_id_npwp, validate_id_npwp
    from .clean_ie_pps import clean_ie_pps, validate_ie_pps
    from .clean_ie_vat import clean_ie_vat, validate_ie_vat
    from .clean_il_hp import clean_il_hp, validate_il_hp
    from .clean_il_idnr import clean_il_idnr, validate_il_idnr
    from .clean_imei import clean_imei, validate_imei
    from .clean_imo import clean_imo, validate_imo
    from .clean_imsi import clean_imsi, validate_imsi
    from .clean_in_aadhaar import clean_in_aadhaar, validate_in_aadhaar
    from .clean_in_epic import clean_in_epic, validate_in_epic
    from .clean_in_gstin import clean_in_gstin, validate_in_gstin
    from .clean_in_pan import clean_in_pan, validate_in_pan
    from .clean_in_vid import clean_in_vid, validate_in_vid
    from .clean_ip import clean_ip, validate_ip
    from .clean_is_kennitala import clean_is_kennitala, validate_is_kennitala
    from .clean_is_vsk import clean_is_vsk, validate_is_vsk
    from .clean_isan import clean_isan, validate_isan
    from .clean_isbn import clean_isbn, validate_isbn
    from .clean_isil import clean_isil, validate_isil
    from .clean_isin import clean_isin, validate_isin
    from .clean_ismn import clean_ismn, validate_ismn
    from .clean_isni import clean_isni, validate_isni
    from .clean_iso6346 import clean_iso6346, validate_iso6346
    from .clean_iso11649 import clean_iso11649, validate_iso11649
    from .clean_isrc import clean_isrc, validate_isrc
    from .clean_issn import clean_issn, validate_issn
    from .clean_it_aic import clean_it_aic, validate_it_aic
    from .clean_it_codicefiscale import clean_it_codicefiscale, validate_it_codicefiscale
    from .clean_it_iva import clean_it_iva, validate_it_iva
    from .clean_jp_cn import clean_jp_cn, validate_jp_cn
    from .clean_jp_in import clean_jp_in, validate_jp_in
    from .clean_json import clean_json, validate_json
    from .clean_ke_pin import clean_ke_pin, validate_ke_pin
    from .clean_kr_brn import clean_kr_brn, validate_kr_brn
    from .clean_kr_rrn import clean_kr_rrn, validate_kr_rrn
    from .clean_lat_long import clean_lat_long, validate_lat_long
    from .clean_lei import clean_lei, validate_lei
    from .clean_li_peid import clean_li_peid, validate_li_peid
    from .clean_lt_asmens import clean_lt_asmens, validate_lt_asmens
    from .clean_lt_pvm import clean_lt_pvm, validate_lt_pvm
    from .clean_lu_tva import clean_lu_tva, validate_lu_tva
    from .clean_lv_pvn import clean_lv_pvn, validate_lv_pvn
    from .clean_ma_ice import clean_ma_ice, validate_ma_ice
    from .clean_mac import clean_mac, validate_mac
    from .clean_mc_tva import clean_mc_tva, validate_mc_tva
    from .clean_md_idno import clean_md_idno, validate_md_idno
    from .clean_me_iban import clean_me_iban, validate_me_iban
    from .clean_me_pib import clean_me_pib, validate_me_pib
    from .clean_meid import clean_meid, validate_meid
    from .clean_mk_edb import clean_mk_edb, validate_mk_edb
    from .clean_ml import clean_ml
    from .clean_mt_vat import clean_mt_vat, validate_mt_vat
    from .clean_mu_nid import clean_mu_nid, validate_mu_nid
    from .clean_mx_curp import clean_mx_curp, validate_mx_curp
    from .clean_mx_rfc import clean_mx_rfc, validate_mx_rfc
    from .clean_my_nric import clean_my_nric, validate_my_nric
    from .clean_mz_nuit import clean_mz_nuit, validate_mz_nuit
    from .clean_nl_brin import clean_nl_brin, validate_nl_brin
    from .clean_nl_bsn import clean_nl_bsn, validate_nl_bsn
    from .clean_nl_btw import clean_nl_btw, validate_nl_btw
    from .clean_nl_identiteitskaartnummer import (
        clean_nl_identiteitskaartnummer,
        validate_nl_identiteitskaartnummer,
    )
    from .clean_nl_onderwijsnummer import clean_nl_onderwijsnummer, validate_nl_onderwijsnummer
    from .clean_nl_postcode import clean_nl_postcode, validate_nl_postcode
    from .clean_no_fodselsnummer import clean_no_fodselsnummer, validate_no_fodselsnummer
    from .clean_no_iban import clean_no_iban, validate_no_iban
    from .clean_no_kontonr import clean_no_kontonr, validate_no_kontonr
    from .clean_no_mva import clean_no_mva, validate_no_mva
    from .clean_no_orgnr import clean_no_orgnr, validate_no_orgnr
    from .clean_nz_bankaccount import clean_nz_bankaccount, validate_nz_bankaccount
    from .clean_nz_ird import clean_nz_ird, validate_nz_ird
    from .clean_pe_cui import clean_pe_cui, validate_pe_cui
    from .clean_pe_ruc import clean_pe_ruc, validate_pe_ruc
    from .clean_phone import clean_phone, validate_phone
    from .clean_pk_cnic import clean_pk_cnic, validate_pk_cnic
    from .clean_pl_nip import clean_pl_nip, validate_pl_nip
    from .clean_pl_pesel import clean_pl_pesel, validate_pl_pesel
    from .clean_pl_regon import clean_pl_regon, validate_pl_regon
    from .clean_pt_cc import clean_pt_cc, validate_pt_cc
    from .clean_pt_nif import clean_pt_nif, validate_pt_nif
    from .clean_py_ruc import clean_py_ruc, validate_py_ruc
    from .clean_ro_cf import clean_ro_cf, validate_ro_cf
    from .clean_ro_cnp import clean_ro_cnp, validate_ro_cnp
    from .clean_ro_cui import clean_ro_cui, validate_ro_cui
    from .clean_ro_onrc import clean_ro_onrc, validate_ro_onrc
    from .clean_rs_pib import clean_rs_pib, validate_rs_pib
    from .clean_ru_inn import clean_ru_inn, validate_ru_inn
    from .clean_ru_ogrn import clean_ru_ogrn, validate_ru_ogrn
    from .clean_se_orgnr import clean_se_orgnr, validate_se_orgnr
    from .clean_se_personnummer import clean_se_personnummer, validate_se_personnummer
    from .clean_se_postnummer import clean_se_postnummer, validate_se_postnummer
    from .clean_se_vat import clean_se_vat, validate_se_vat
    from .clean_sg_uen import clean_sg_uen, validate_sg_uen
    from .clean_si_ddv import clean_si_ddv, validate_si_ddv
    from .clean_si_emso import clean_si_emso, validate_si_emso
    from .clean_si_maticna import clean_si_maticna, validate_si_maticna
    from .clean_sk_dph import clean_sk_dph, validate_sk_dph
    from .clean_sk_rc import clean_sk_rc, validate_sk_rc
    from .clean_sm_coe import clean_sm_coe, validate_sm_coe
    from .clean_sn_ninea import clean_sn_ninea, validate_sn_ninea
    from .clean_sv_nit import clean_sv_nit, validate_sv_nit
    from .clean_text import clean_text, default_text_pipeline
    from .clean_th_moa import clean_th_moa, validate_th_moa
    from .clean_th_pin import clean_th_pin, validate_th_pin
    from .clean_th_tin import clean_th_tin, validate_th_tin
    from .clean_tn_mf import clean_tn_mf, validate_tn_mf
    from .clean_tr_tckimlik import clean_tr_tckimlik, validate_tr_tckimlik
    from .clean_tr_vkn import clean_tr_vkn, validate_tr_vkn
    from .clean_tw_ubn import clean_tw_ubn, validate_tw_ubn
    from .clean_ua_edrpou import clean_ua_edrpou, validate_ua_edrpou
    from .clean_ua_rntrc import clean_ua_rntrc, validate_ua_rntrc
    from .clean_url import clean_url, validate_url
    from .clean_us_atin import clean_us_atin, validate_us_atin
    from .clean_us_ein import clean_us_ein, validate_us_ein
    from .clean_us_itin import clean_us_itin, validate_us_itin
    from .clean_us_ptin import clean_us_ptin, validate_us_ptin
    from .clean_us_rtn import clean_us_rtn, validate_us_rtn
    from .clean_us_ssn import clean_us_ssn, validate_us_ssn
    from .clean_us_tin import clean_us_tin, validate_us_tin
    from .clean_uy_rut import clean_uy_rut, validate_uy_rut
    from .clean_vatin import clean_vatin, validate_vatin
    from .clean_ve_rif import clean_ve_rif, validate_ve_rif
    from .clean_vn_mst import clean_vn_mst, validate_vn_mst
    from .clean_za_idnr import clean_za_idnr, validate_za_idnr
    from .clean_za_tin import clean_za_tin, validate_za_tin

__all__ = [
    "clean_lat_long",
    "validate_lat_long",
    "clean_email",
    "validate_email",
    "clean_country",
    "validate_country",
    "clean_url",
    "validate_url",
    "clean_phone",
    "validate_phone",
    "clean_json",
    "validate_json",
    "clean_ip",
    "validate_ip",
    "clean_headers",
    "clean_address",
    "validate_address",
    "clean_date",
    "validate_date",
    "clean_duplication",
    "clean_currency",
    "validate_currency",
    "clean_df",
    "clean_df_gui",
    "clean_text",
    "default_text_pipeline",
    "clean_au_abn",
    "validate_au_abn",
    "clean_au_acn",
    "validate_au_acn",
    "clean_au_tfn",
    "validate_au_tfn",
    "clean_be_iban",
    "validate_be_iban",
    "clean_be_vat",
    "validate_be_vat",
    "clean_bg_egn",
    "validate_bg_egn",
    "clean_bg_pnf",
    "validate_bg_pnf",
    "clean_bg_vat",
    "validate_bg_vat",
    "clean_br_cnpj",
    "validate_br_cnpj",
    "clean_br_cpf",
    "validate_br_cpf",
    "clean_by_unp",
    "validate_by_unp",
    "clean_ca_bn",
    "validate_ca_bn",
    "clean_ca_sin",
    "validate_ca_sin",
    "clean_ch_esr",
    "validate_ch_esr",
    "clean_ch_ssn",
    "validate_ch_ssn",
    "clean_ch_uid",
    "validate_ch_uid",
    "clean_ch_vat",
    "validate_ch_vat",
    "clean_cl_rut",
    "validate_cl_rut",
    "clean_cn_ric",
    "validate_cn_ric",
    "clean_cn_uscc",
    "validate_cn_uscc",
    "clean_co_nit",
    "validate_co_nit",
    "clean_cr_cpf",
    "validate_cr_cpf",
    "clean_cr_cpj",
    "validate_cr_cpj",
    "clean_cr_cr",
    "validate_cr_cr",
    "clean_cu_ni",
    "validate_cu_ni",
    "clean_cy_vat",
    "validate_cy_vat",
    "clean_cz_dic",
    "validate_cz_dic",
    "clean_cz_rc",
    "validate_cz_rc",
    "clean_de_handelsregisternummer",
    "validate_de_handelsregisternummer",
    "clean_de_idnr",
    "validate_de_idnr",
    "clean_de_stnr",
    "validate_de_stnr",
    "clean_de_vat",
    "validate_de_vat",
    "clean_de_wkn",
    "validate_de_wkn",
    "clean_dk_cpr",
    "validate_dk_cpr",
    "clean_dk_cvr",
    "validate_dk_cvr",
    "clean_do_cedula",
    "validate_do_cedula",
    "clean_do_ncf",
    "validate_do_ncf",
    "clean_do_rnc",
    "validate_do_rnc",
    "clean_ec_ci",
    "validate_ec_ci",
    "clean_ec_ruc",
    "validate_ec_ruc",
    "clean_ee_ik",
    "validate_ee_ik",
    "clean_ee_kmkr",
    "validate_ee_kmkr",
    "clean_ee_registrikood",
    "validate_ee_registrikood",
    "clean_es_ccc",
    "validate_es_ccc",
    "clean_es_cif",
    "validate_es_cif",
    "clean_es_cups",
    "validate_es_cups",
    "clean_es_dni",
    "validate_es_dni",
    "clean_es_iban",
    "validate_es_iban",
    "clean_es_nie",
    "validate_es_nie",
    "clean_es_nif",
    "validate_es_nif",
    "clean_es_referenciacatastral",
    "validate_es_referenciacatastral",
    "clean_eu_at_02",
    "validate_eu_at_02",
    "clean_eu_banknote",
    "validate_eu_banknote",
    "clean_eu_eic",
    "validate_eu_eic",
    "clean_eu_nace",
    "validate_eu_nace",
    "clean_eu_vat",
    "validate_eu_vat",
    "clean_fi_alv",
    "validate_fi_alv",
    "clean_fi_associationid",
    "validate_fi_associationid",
    "clean_fi_hetu",
    "validate_fi_hetu",
    "clean_fi_veronumero",
    "validate_fi_veronumero",
    "clean_fi_ytunnus",
    "validate_fi_ytunnus",
    "clean_fr_nif",
    "validate_fr_nif",
    "clean_fr_nir",
    "validate_fr_nir",
    "clean_fr_siren",
    "validate_fr_siren",
    "clean_fr_siret",
    "validate_fr_siret",
    "clean_fr_tva",
    "validate_fr_tva",
    "clean_gb_nhs",
    "validate_gb_nhs",
    "clean_gb_sedol",
    "validate_gb_sedol",
    "clean_gb_upn",
    "validate_gb_upn",
    "clean_gb_utr",
    "validate_gb_utr",
    "clean_gb_vat",
    "validate_gb_vat",
    "clean_gr_amka",
    "validate_gr_amka",
    "clean_gr_vat",
    "validate_gr_vat",
    "clean_gt_nit",
    "validate_gt_nit",
    "clean_hr_oib",
    "validate_hr_oib",
    "clean_hu_anum",
    "validate_hu_anum",
    "clean_id_npwp",
    "validate_id_npwp",
    "clean_ie_pps",
    "validate_ie_pps",
    "clean_ie_vat",
    "validate_ie_vat",
    "clean_il_hp",
    "validate_il_hp",
    "clean_il_idnr",
    "validate_il_idnr",
    "clean_in_aadhaar",
    "validate_in_aadhaar",
    "clean_in_pan",
    "validate_in_pan",
    "clean_is_kennitala",
    "validate_is_kennitala",
    "clean_is_vsk",
    "validate_is_vsk",
    "clean_it_aic",
    "validate_it_aic",
    "clean_it_codicefiscale",
    "validate_it_codicefiscale",
    "clean_it_iva",
    "validate_it_iva",
    "clean_jp_cn",
    "validate_jp_cn",
    "clean_kr_brn",
    "validate_kr_brn",
    "clean_kr_rrn",
    "validate_kr_rrn",
    "clean_li_peid",
    "validate_li_peid",
    "clean_lt_asmens",
    "validate_lt_asmens",
    "clean_lt_pvm",
    "validate_lt_pvm",
    "clean_lu_tva",
    "validate_lu_tva",
    "clean_lv_pvn",
    "validate_lv_pvn",
    "clean_mc_tva",
    "validate_mc_tva",
    "clean_md_idno",
    "validate_md_idno",
    "clean_me_iban",
    "validate_me_iban",
    "clean_mt_vat",
    "validate_mt_vat",
    "clean_mu_nid",
    "validate_mu_nid",
    "clean_mx_curp",
    "validate_mx_curp",
    "clean_mx_rfc",
    "validate_mx_rfc",
    "clean_my_nric",
    "validate_my_nric",
    "clean_nl_brin",
    "validate_nl_brin",
    "clean_nl_bsn",
    "validate_nl_bsn",
    "clean_nl_btw",
    "validate_nl_btw",
    "clean_nl_onderwijsnummer",
    "validate_nl_onderwijsnummer",
    "clean_nl_postcode",
    "validate_nl_postcode",
    "clean_no_fodselsnummer",
    "validate_no_fodselsnummer",
    "clean_no_iban",
    "validate_no_iban",
    "clean_no_kontonr",
    "validate_no_kontonr",
    "clean_no_mva",
    "validate_no_mva",
    "clean_no_orgnr",
    "validate_no_orgnr",
    "clean_nz_bankaccount",
    "validate_nz_bankaccount",
    "clean_nz_ird",
    "validate_nz_ird",
    "clean_pe_cui",
    "validate_pe_cui",
    "clean_pe_ruc",
    "validate_pe_ruc",
    "clean_pl_nip",
    "validate_pl_nip",
    "clean_pl_pesel",
    "validate_pl_pesel",
    "clean_pl_regon",
    "validate_pl_regon",
    "clean_pt_nif",
    "validate_pt_nif",
    "clean_py_ruc",
    "validate_py_ruc",
    "clean_ro_cf",
    "validate_ro_cf",
    "clean_ro_cnp",
    "validate_ro_cnp",
    "clean_ro_cui",
    "validate_ro_cui",
    "clean_ro_onrc",
    "validate_ro_onrc",
    "clean_isbn",
    "validate_isbn",
    "clean_bic",
    "validate_bic",
    "clean_bitcoin",
    "validate_bitcoin",
    "clean_casrn",
    "validate_casrn",
    "clean_cusip",
    "validate_cusip",
    "clean_ean",
    "validate_ean",
    "clean_figi",
    "validate_figi",
    "clean_grid",
    "validate_grid",
    "clean_iban",
    "validate_iban",
    "clean_imei",
    "validate_imei",
    "clean_imo",
    "validate_imo",
    "clean_imsi",
    "validate_imsi",
    "clean_isan",
    "validate_isan",
    "clean_isil",
    "validate_isil",
    "clean_isin",
    "validate_isin",
    "clean_ismn",
    "validate_ismn",
    "clean_issn",
    "validate_issn",
    "clean_ad_nrt",
    "validate_ad_nrt",
    "clean_al_nipt",
    "validate_al_nipt",
    "clean_ar_cbu",
    "validate_ar_cbu",
    "clean_ar_cuit",
    "validate_ar_cuit",
    "clean_ar_dni",
    "validate_ar_dni",
    "clean_at_uid",
    "validate_at_uid",
    "clean_at_vnr",
    "validate_at_vnr",
    "clean_lei",
    "validate_lei",
    "clean_meid",
    "validate_meid",
    "clean_vatin",
    "validate_vatin",
    "clean_ml",
    "clean_at_businessid",
    "validate_at_businessid",
    "clean_at_postleitzahl",
    "validate_at_postleitzahl",
    "clean_at_tin",
    "validate_at_tin",
    "clean_az_voen",
    "validate_az_voen",
    "clean_be_bis",
    "validate_be_bis",
    "clean_be_eid",
    "validate_be_eid",
    "clean_be_nn",
    "validate_be_nn",
    "clean_be_ogm_vcs",
    "validate_be_ogm_vcs",
    "clean_be_ssn",
    "validate_be_ssn",
    "clean_ca_bc_phn",
    "validate_ca_bc_phn",
    "clean_cfi",
    "validate_cfi",
    "clean_cz_bankaccount",
    "validate_cz_bankaccount",
    "clean_de_leitweg",
    "validate_de_leitweg",
    "clean_dz_nif",
    "validate_dz_nif",
    "clean_eg_tn",
    "validate_eg_tn",
    "clean_es_cae",
    "validate_es_cae",
    "clean_es_postal_code",
    "validate_es_postal_code",
    "clean_eu_ecnumber",
    "validate_eu_ecnumber",
    "clean_eu_excise",
    "validate_eu_excise",
    "clean_eu_oss",
    "validate_eu_oss",
    "clean_fo_vn",
    "validate_fo_vn",
    "clean_fr_accise",
    "validate_fr_accise",
    "clean_fr_rcs",
    "validate_fr_rcs",
    "clean_gh_tin",
    "validate_gh_tin",
    "clean_gn_nifp",
    "validate_gn_nifp",
    "clean_gs1_128",
    "validate_gs1_128",
    "clean_id_nik",
    "validate_id_nik",
    "clean_in_epic",
    "validate_in_epic",
    "clean_in_gstin",
    "validate_in_gstin",
    "clean_in_vid",
    "validate_in_vid",
    "clean_isni",
    "validate_isni",
    "clean_iso11649",
    "validate_iso11649",
    "clean_iso6346",
    "validate_iso6346",
    "clean_isrc",
    "validate_isrc",
    "clean_jp_in",
    "validate_jp_in",
    "clean_ke_pin",
    "validate_ke_pin",
    "clean_ma_ice",
    "validate_ma_ice",
    "clean_mac",
    "validate_mac",
    "clean_me_pib",
    "validate_me_pib",
    "clean_mk_edb",
    "validate_mk_edb",
    "clean_mz_nuit",
    "validate_mz_nuit",
    "clean_nl_identiteitskaartnummer",
    "validate_nl_identiteitskaartnummer",
    "clean_pk_cnic",
    "validate_pk_cnic",
    "clean_pt_cc",
    "validate_pt_cc",
    "clean_rs_pib",
    "validate_rs_pib",
    "clean_ru_inn",
    "validate_ru_inn",
    "clean_ru_ogrn",
    "validate_ru_ogrn",
    "clean_se_orgnr",
    "validate_se_orgnr",
    "clean_se_personnummer",
    "validate_se_personnummer",
    "clean_se_postnummer",
    "validate_se_postnummer",
    "clean_se_vat",
    "validate_se_vat",
    "clean_sg_uen",
    "validate_sg_uen",
    "clean_si_ddv",
    "validate_si_ddv",
    "clean_si_emso",
    "validate_si_emso",
    "clean_si_maticna",
    "validate_si_maticna",
    "clean_sk_dph",
    "validate_sk_dph",
    "clean_sk_rc",
    "validate_sk_rc",
    "clean_sm_coe",
    "validate_sm_coe",
    "clean_sn_ninea",
    "validate_sn_ninea",
    "clean_sv_nit",
    "validate_sv_nit",
    "clean_th_moa",
    "validate_th_moa",
    "clean_th_pin",
    "validate_th_pin",
    "clean_th_tin",
    "validate_th_tin",
    "clean_tn_mf",
    "validate_tn_mf",
    "clean_tr_tckimlik",
    "validate_tr_tckimlik",
    "clean_tr_vkn",
    "validate_tr_vkn",
    "clean_tw_ubn",
    "validate_tw_ubn",
    "clean_ua_edrpou",
    "validate_ua_edrpou",
    "clean_ua_rntrc",
    "validate_ua_rntrc",
    "clean_us_atin",
    "validate_us_atin",
    "clean_us_ein",
    "validate_us_ein",
    "clean_us_itin",
    "validate_us_itin",
    "clean_us_ptin",
    "validate_us_ptin",
    "clean_us_rtn",
    "validate_us_rtn",
    "clean_us_ssn",
    "validate_us_ssn",
    "clean_us_tin",
    "validate_us_tin",
    "clean_uy_rut",
    "validate_uy_rut",
    "clean_ve_rif",
    "validate_ve_rif",
    "clean_vn_mst",
    "validate_vn_mst",
    "clean_za_idnr",
    "validate_za_idnr",
    "clean_za_tin",
    "validate_za_tin",
]

# Public names that do not live in the module suggested by their own name
# (clean_<x> and validate_<x> are both defined in the module clean_<x>).
_MODULE_OVERRIDES = {"default_text_pipeline": "clean_text"}
_PUBLIC = frozenset(__all__)


def _module_of(name: str) -> str:
    """Return the name of the submodule that defines the public function ``name``."""
    if name in _MODULE_OVERRIDES:
        return _MODULE_OVERRIDES[name]
    if name.startswith("validate_"):
        return "clean_" + name[len("validate_") :]
    return name


def __getattr__(name: str) -> Any:
    """Import the public functions, and the submodules of this package, on first access."""
    if name in _PUBLIC:
        module = importlib.import_module(f"{__name__}.{_module_of(name)}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        full_name = f"{__name__}.{name}"
        try:
            return importlib.import_module(full_name)
        except ModuleNotFoundError as err:
            if err.name != full_name:
                raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PUBLIC)


class _CleanPackage(types.ModuleType):
    """
    Module type of this package.

    Most cleaning modules are named after the function they define (the module
    ``dataprep.clean.clean_email`` defines the function ``clean_email``). When a
    submodule is imported, the import system binds it as an attribute of the
    package, which would hide the function of the same name. This class makes
    sure that ``dataprep.clean.clean_email`` is always the function, no matter
    whether the function or its module was imported first. The modules stay
    importable by their full name (``from dataprep.clean.clean_email import ...``).
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in _PUBLIC
            and isinstance(value, types.ModuleType)
            and value.__name__ == f"{self.__name__}.{name}"
        ):
            value = getattr(value, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CleanPackage
