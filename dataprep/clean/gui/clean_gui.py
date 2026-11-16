"""
Flask backend of the Dataprep.Clean GUI.

The backend serves the pre-built frontend in ``frontend_dist`` and a small JSON
API that applies the clean functions to the DataFrame being cleaned. It is
started by :func:`dataprep.clean.clean_df_gui`; flask is only imported when
this module is imported.
"""

# pylint: disable=R0912, R0915, invalid-name, too-many-lines
from __future__ import annotations

import importlib
import inspect
import logging
import socket
import threading
from ast import literal_eval
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    from flask import Flask, Response, render_template, request
    from flask_cors import CORS
    from werkzeug.serving import BaseWSGIServer, make_server
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "clean_df_gui requires the flask and flask-cors packages. "
        "Install them with: pip install flask flask-cors"
    ) from err

from ..utils import apply_partitions

DEFAULT_PORT = 7680

# instantiate the app
app = Flask(__name__, static_folder="frontend_dist/static", template_folder="frontend_dist")

# enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# disable useless log
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class _CleanFunctions:
    """
    Dictionary-like access to the clean functions by name. The functions are
    looked up in dataprep.clean on demand, so that only the functions that are
    actually used in the GUI are imported.
    """

    def __getitem__(self, name: str) -> Callable[..., Any]:
        if name not in clean_list:
            raise KeyError(name)
        func: Callable[..., Any] = getattr(importlib.import_module("dataprep.clean"), name)
        return func

    def __contains__(self, name: object) -> bool:
        return name in clean_list


clean_list = [
    "clean_email",
    "clean_country",
    "clean_ip",
    "clean_phone",
    "clean_url",
    "clean_headers",
    "clean_date",
    "clean_lat_long",
    "clean_json",
    "clean_address",
    "clean_df",
    "clean_duplication",
    "clean_currency",
    "clean_au_abn",
    "clean_au_acn",
    "clean_au_tfn",
    "clean_be_iban",
    "clean_be_vat",
    "clean_bg_egn",
    "clean_bg_pnf",
    "clean_bg_vat",
    "clean_br_cnpj",
    "clean_br_cpf",
    "clean_by_unp",
    "clean_ca_bn",
    "clean_ca_sin",
    "clean_ch_esr",
    "clean_ch_ssn",
    "clean_ch_uid",
    "clean_ch_vat",
    "clean_cl_rut",
    "clean_cn_ric",
    "clean_cn_uscc",
    "clean_co_nit",
    "clean_cr_cpf",
    "clean_cr_cpj",
    "clean_cr_cr",
    "clean_cu_ni",
    "clean_cy_vat",
    "clean_cz_dic",
    "clean_cz_rc",
    "clean_de_handelsregisternummer",
    "clean_de_idnr",
    "clean_de_stnr",
    "clean_de_vat",
    "clean_de_wkn",
    "clean_dk_cpr",
    "clean_dk_cvr",
    "clean_do_cedula",
    "clean_do_ncf",
    "clean_do_rnc",
    "clean_ec_ci",
    "clean_ec_ruc",
    "clean_ee_ik",
    "clean_ee_kmkr",
    "clean_ee_registrikood",
    "clean_es_ccc",
    "clean_es_cif",
    "clean_es_cups",
    "clean_es_dni",
    "clean_es_iban",
    "clean_es_nie",
    "clean_es_nif",
    "clean_es_referenciacatastral",
    "clean_eu_at_02",
    "clean_eu_banknote",
    "clean_eu_eic",
    "clean_eu_nace",
    "clean_eu_vat",
    "clean_fi_alv",
    "clean_fi_associationid",
    "clean_fi_hetu",
    "clean_fi_veronumero",
    "clean_fi_ytunnus",
    "clean_fr_nif",
    "clean_fr_nir",
    "clean_fr_siren",
    "clean_fr_siret",
    "clean_fr_tva",
    "clean_gb_nhs",
    "clean_gb_sedol",
    "clean_gb_upn",
    "clean_gb_utr",
    "clean_gb_vat",
    "clean_gr_amka",
    "clean_gr_vat",
    "clean_gt_nit",
    "clean_hr_oib",
    "clean_hu_anum",
    "clean_id_npwp",
    "clean_ie_pps",
    "clean_ie_vat",
    "clean_il_hp",
    "clean_il_idnr",
    "clean_in_aadhaar",
    "clean_in_pan",
    "clean_is_kennitala",
    "clean_is_vsk",
    "clean_it_aic",
    "clean_it_codicefiscale",
    "clean_it_iva",
    "clean_jp_cn",
    "clean_kr_brn",
    "clean_kr_rrn",
    "clean_li_peid",
    "clean_lt_pvm",
    "clean_lt_asmens",
    "clean_lu_tva",
    "clean_lv_pvn",
    "clean_mc_tva",
    "clean_md_idno",
    "clean_me_iban",
    "clean_mt_vat",
    "clean_mu_nid",
    "clean_mx_curp",
    "clean_mx_rfc",
    "clean_my_nric",
    "clean_nl_brin",
    "clean_nl_btw",
    "clean_nl_bsn",
    "clean_nl_onderwijsnummer",
    "clean_nl_postcode",
    "clean_no_fodselsnummer",
    "clean_no_iban",
    "clean_no_kontonr",
    "clean_no_mva",
    "clean_no_orgnr",
    "clean_nz_bankaccount",
    "clean_nz_ird",
    "clean_pe_cui",
    "clean_pe_ruc",
    "clean_pl_nip",
    "clean_pl_pesel",
    "clean_pl_regon",
    "clean_pt_nif",
    "clean_py_ruc",
    "clean_ro_cf",
    "clean_ro_cnp",
    "clean_ro_cui",
    "clean_ro_onrc",
    "clean_isbn",
    "clean_bic",
    "clean_bitcoin",
    "clean_casrn",
    "clean_cusip",
    "clean_ean",
    "clean_figi",
    "clean_grid",
    "clean_iban",
    "clean_imei",
    "clean_imo",
    "clean_imsi",
    "clean_isan",
    "clean_isil",
    "clean_isin",
    "clean_ismn",
    "clean_issn",
    "clean_ad_nrt",
    "clean_al_nipt",
    "clean_ar_cbu",
    "clean_ar_cuit",
    "clean_ar_dni",
    "clean_at_uid",
    "clean_at_vnr",
    "clean_lei",
    "clean_meid",
    "clean_vatin",
]
# Dictionary of functions
clean_function_dic = _CleanFunctions()

contain_birthday_place = ["clean_cn_ric", "clean_my_nric"]
contain_isin = ["clean_de_wkn", "clean_gb_sedol", "clean_cusip"]
contain_mask = ["clean_in_aadhaar", "clean_in_pan"]
contain_iban = ["clean_no_kontonr", "clean_es_ccc"]
contain_gender = [
    "clean_cu_ni",
    "clean_ee_ik",
    "clean_gr_amka",
    "clean_mx_curp",
    "clean_it_codicefiscale",
    "clean_no_fodselsnummer",
    "clean_pl_pesel",
]
contain_birthdate = [
    "clean_bg_egn",
    "clean_dk_cpr",
    "clean_kr_rrn",
    "clean_mu_nid",
    "clean_lt_asmens",
    "clean_lv_pvn",
    "clean_ro_cnp",
]
contain_split = [
    "clean_bic",
    "clean_bitcoin",
    "clean_casrn",
    "clean_cusip",
    "clean_ean",
    "clean_figi",
    "clean_grid",
    "clean_iban",
    "clean_imei",
    "clean_imo",
    "clean_imsi",
    "clean_isan",
    "clean_isil",
    "clean_isin",
    "clean_ismn",
    "clean_issn",
    "clean_ad_nrt",
    "clean_al_nipt",
    "clean_ar_cbu",
    "clean_ar_cuit",
    "clean_ar_dni",
    "clean_at_uid",
    "clean_at_vnr",
    "clean_lei",
    "clean_meid",
    "clean_vatin",
]


class GuiState:
    """
    The state of the GUI: the DataFrame being cleaned, the history of the
    cleaning steps (for undo and redo) and the log of the applied operations.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.origin_df = pd.DataFrame({})
        self.history: List[pd.DataFrame] = [self.origin_df]
        self.logs: List[str] = []
        self.pointer = 0
        self.exported_df: Optional[pd.DataFrame] = None
        self.exported_log: str = ""

    def reset(self, df: pd.DataFrame) -> None:
        """Start cleaning a new DataFrame, forgetting all previous steps."""
        with self.lock:
            self.origin_df = df
            self.history = [df]
            self.logs = []
            self.pointer = 0

    @property
    def df(self) -> pd.DataFrame:
        """The current version of the DataFrame."""
        return self.history[self.pointer]

    @property
    def operation_log(self) -> List[str]:
        """The operations that lead from the original to the current DataFrame."""
        return self.logs[: self.pointer]

    def push(self, df: pd.DataFrame, log: str) -> None:
        """Record a new cleaning step. Steps that were undone are discarded."""
        with self.lock:
            self.history = self.history[: self.pointer + 1] + [df]
            self.logs = self.logs[: self.pointer] + [log]
            self.pointer += 1

    def undo(self) -> None:
        """Go back one step."""
        with self.lock:
            self.pointer = max(self.pointer - 1, 0)

    def redo(self) -> None:
        """Redo the step that was undone last."""
        with self.lock:
            self.pointer = min(self.pointer + 1, len(self.history) - 1)

    def code(self) -> str:
        """Python code that reproduces the current DataFrame from the original one."""
        return "".join(f"df = {log}\n" for log in self.operation_log)


STATE = GuiState()


def _to_bool(value: Any) -> bool:
    """Convert a form value such as "True", "true" or "1" to a boolean."""
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _table_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """
    The JSON description of a DataFrame expected by the frontend. Every value is
    sent as a string, so that the response is valid JSON whatever the data types
    and missing values of the DataFrame are.
    """
    col_names = [str(col) for col in df.columns]
    table_columns = [
        {"colName": col_name, "colLabel": col_name, "colWidth": 180} for col_name in col_names
    ]
    columns = [[str(val) for val in df.iloc[:, pos].tolist()] for pos in range(df.shape[1])]
    table_data = [dict(zip(col_names, row)) for row in zip(*columns)]
    return {"tableData": table_data, "tableColumns": table_columns}


# sanity check route
@app.route("/", methods=["GET"])
def index() -> Any:
    """
    This function shows the index of GUI.
    """
    return render_template("index.html")


@app.route("/getInitSheet", methods=["GET"])
def getInitSheet() -> Any:
    """
    This function shows the input table in GUI.
    """
    return _table_payload(STATE.df)


@app.route("/fileUpload", methods=["POST"])
def getUploadedFile() -> Any:
    """
    This function shows the uploaded table file in GUI.
    """
    file = request.files.get("file")
    if not file:
        return "No File"
    STATE.reset(pd.read_csv(file.stream))
    return _table_payload(STATE.df)


@app.route("/cleanWholeDF", methods=["POST"])
def cleanWholeDF() -> Any:
    """
    This function receive the request of clean whole dataframe.
    """
    params = request.form
    clean_headers = _to_bool(params.get("clean_headers", True))
    data_type_detection = params.get("data_type_detection", "semantic")
    standardize_missing = params.get("standardize_missing_values", "fill")
    downcast_memory = _to_bool(params.get("downcast_memory", True))

    try:
        result = clean_function_dic["clean_df"](
            STATE.df,
            clean_header=clean_headers,
            data_type_detection=data_type_detection,
            standardize_missing_values=standardize_missing,
            downcast_memory=downcast_memory,
            report=False,
        )
    except (TypeError, KeyError, ValueError) as error:
        return {"error": f"clean_df failed: {error}"}, 400
    # clean_df returns (data types, DataFrame) unless data_type_detection is "none"
    df_cleaned = result[1] if isinstance(result, tuple) else result

    # Update log with clean_df function
    log = (
        f"clean_df(df, clean_header={clean_headers}, "
        f"data_type_detection='{data_type_detection}', "
        f"standardize_missing_values='{standardize_missing}', "
        f"downcast_memory={downcast_memory}, report=False)"
    )
    if data_type_detection != "none":
        log += "[1]"
    STATE.push(df_cleaned, log)

    return _table_payload(df_cleaned)


@app.route("/getOperationLog", methods=["GET"])
def getOperationLog() -> Any:
    """
    This function receive the request of get log of operations.
    """
    return {"operationLog": STATE.operation_log}


@app.route("/getFunctionParams", methods=["GET", "POST"])
def getFunctionParams() -> Any:
    """
    This function receive the request of get parameter of clean functions.
    """
    col_names = [str(col) for col in STATE.df.columns]
    table_columns = []
    for col_name in col_names:
        temp_dic = {}
        temp_dic["value"] = col_name
        temp_dic["label"] = col_name
        table_columns.append(temp_dic)

    param_dic = {}
    param_default = {}
    info = request.get_json(silent=True) or {}
    clean_func = info.get("clean_func")
    if clean_func not in clean_list:
        return {"error": f"unknown clean function {clean_func}"}, 400

    args = inspect.signature(clean_function_dic[clean_func]).parameters
    args_list = list(args.keys())

    for arg in args_list:
        temp_option_list = []
        if arg in ("df", "column"):
            continue
        if arg == "inplace":
            break

        if clean_func == "clean_email":
            if isinstance(args[arg].default, bool):
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_headers":
            if arg == "case":
                temp_option_list.append({"value": "snake", "label": "Snake ('column_name')"})
                temp_option_list.append({"value": "kebab", "label": "Kebab ('column-name')"})
                temp_option_list.append({"value": "camel", "label": "Camel ('columnName')"})
                temp_option_list.append({"value": "pascal", "label": "Pascal ('ColumnName')"})
                temp_option_list.append({"value": "const", "label": "Const ('COLUMN_NAME')"})
                temp_option_list.append({"value": "sentence", "label": "Sentence ('Column name')"})
                temp_option_list.append({"value": "title", "label": "Title ('Column Name')"})
                temp_option_list.append({"value": "lower", "label": "Lower ('column name')"})
                temp_option_list.append({"value": "upper", "label": "Upper ('COLUMN NAME')"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "remove_accents":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_country":
            if arg == "input_format":
                temp_option_list.append({"value": "auto", "label": "Auto"})
                temp_option_list.append({"value": "name", "label": "Name"})
                temp_option_list.append({"value": "official", "label": "Official"})
                temp_option_list.append({"value": "alpha-2", "label": "Alpha-2"})
                temp_option_list.append({"value": "alpha-3", "label": "Alpha-3"})
                temp_option_list.append({"value": "numeric", "label": "Numeric"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append({"value": "name", "label": "Name"})
                temp_option_list.append({"value": "official", "label": "Official"})
                temp_option_list.append({"value": "alpha-2", "label": "Alpha-2"})
                temp_option_list.append({"value": "alpha-3", "label": "Alpha-3"})
                temp_option_list.append({"value": "numeric", "label": "Numeric"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fuzzy_dist":
                for i in range(0, 11):
                    temp_option_list.append({"value": str(i), "label": str(i)})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "strict":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_date":
            if arg == "output_format":
                temp_option_list.append(
                    {"value": "YYYY-MM-DD hh:mm:ss", "label": "YYYY-MM-DD hh:mm:ss"}
                )
                temp_option_list.append({"value": "YYYY-MM-DD", "label": "YYYY-MM-DD"})
                temp_option_list.append(
                    {"value": "YYYY-MM-DD AD at hh:mm:ss Z", "label": "YYYY-MM-DD AD at hh:mm:ss Z"}
                )
                temp_option_list.append(
                    {"value": "EEE, d MMM yyyy HH:mm:ss Z", "label": "EEE, d MMM yyyy HH:mm:ss Z"}
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fix_missing":
                temp_option_list.append(
                    {
                        "value": "minimum",
                        "label": "Fill hour, minute, second with 0, "
                        "and month, day, year with January, 1st, 2000.",
                    }
                )
                temp_option_list.append(
                    {"value": "current", "label": "Fill with the current date and time"}
                )
                temp_option_list.append({"value": "empty", "label": "None"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "infer_day_first":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_lat_long":
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append(
                    {"value": "dd", "label": "Decimal degrees (51.4934, 0.0098)"}
                )
                temp_option_list.append(
                    {
                        "value": "ddh",
                        "label": "Decimal degrees with hemisphere ('51.4934° N, 0.0098° E')",
                    }
                )
                temp_option_list.append(
                    {"value": "dm", "label": "Degrees minutes ('51° 29.604' N, 0° 0.588' E')"}
                )
                temp_option_list.append(
                    {
                        "value": "dms",
                        "label": "Degrees minutes seconds ('51° 29' 36.24\" N, 0° 0' 35.28\" E')",
                    }
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_ip":
            if arg == "input_format":
                temp_option_list.append({"value": "auto", "label": "Auto"})
                temp_option_list.append({"value": "ipv4", "label": "IPV4"})
                temp_option_list.append({"value": "ipv6", "label": "IPV6"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append({"value": "compressed", "label": "Compressed (12.3.4.5)"})
                temp_option_list.append({"value": "full", "label": "Full ('0012.0003.0004.0005')"})
                temp_option_list.append(
                    {"value": "binary", "label": "Binary ('00001100000000110000010000000101')"}
                )
                temp_option_list.append({"value": "hexa", "label": "Hexa ('0xc030405')"})
                temp_option_list.append({"value": "integer", "label": "Integer (201524229)"})
                temp_option_list.append({"value": "packed", "label": "Packed"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_phone":
            if arg == "output_format":
                temp_option_list.append({"value": "nanp", "label": "nanp ('NPA-NXX-XXXX')"})
                temp_option_list.append({"value": "e164", "label": "e164 ('+1NPANXXXXXX')"})
                temp_option_list.append(
                    {"value": "national", "label": "National ('(NPA) NXX-XXXX)"}
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fix_missing":
                temp_option_list.append({"value": "empty", "label": "Empty"})
                temp_option_list.append({"value": "auto", "label": "Auto"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        # elif clean_func == "clean_text":
        #    pass
        elif clean_func == "clean_url":
            if arg in ["remove_auth", "split"]:
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_address":
            if arg == "output_format":
                temp_option_list.append(
                    {
                        "value": "(building) house_number street_prefix_abbr street_name "
                        "street_suffix_abbr, apartment, city, state_abbr zipcode",
                        "label": "(building) house_number street_prefix_abbr street_name "
                        "street_suffix_abbr, apartment, city, state_abbr zipcode",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "(zipcode) street_prefix_full street_name ~state_full~",
                        "label": "(zipcode) street_prefix_full street_name ~state_full~",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "house_number street_name street_suffix_full (building)",
                        "label": "house_number street_name street_suffix_full (building)",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        # elif clean_func == "clean_duplication":
        #     if arg == "page_size":
        #         for i in range(5, 8):
        #             temp_option_list.append({"value": str(i), "label": str(i)})
        #         param_dic[arg] = temp_option_list
        #         param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_currency":
            if arg == "input_currency":
                temp_option_list.append({"value": "usd", "label": "USD"})
                temp_option_list.append(
                    {
                        "value": "Binance Coin",
                        "label": "Binance Coin",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "ethereum",
                        "label": "Ethereum",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "target_representation":
                temp_option_list.append(
                    {"value": "decimal", "label": "Decimal(floating point number)"}
                )
                temp_option_list.append(
                    {
                        "value": "abbr",
                        "label": "Abbreviation((string with comma seprated values)",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_au_acn":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "abn", "label": "Australian Business Number"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_birthday_place:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})
                temp_option_list.append({"value": "birthplace", "label": "Birthplace"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_gender:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})
                temp_option_list.append({"value": "gender", "label": "Gender"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_isin:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "isin", "label": "ISIN"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_birthdate:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})

            param_dic[arg] = temp_option_list
            param_default[arg] = str(args[arg].default)
        elif clean_func in contain_iban:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "iban", "label": "IBAN"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_es_iban":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "ccc", "label": "Código Cuenta Corriente"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_eu_nace":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "label", "label": "Label"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_eu_vat":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "country", "label": "Country"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_fr_siren":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "tva", "label": "TVA"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_mask:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "mask", "label": "Mask"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_it_aic":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "base10", "label": "BASE10"})
                temp_option_list.append({"value": "base32", "label": "BASE32"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_no_iban":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append(
                    {"value": "kontonr", "label": "Norwegian bank account part"}
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_nz_bankaccount":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "info", "label": "Supplied number"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_pe_cui":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "ruc", "label": "RUC"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_pe_ruc":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "dni", "label": "DNI (CUI)"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_isbn":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "isbn10", "label": "ISBN-10"})
                temp_option_list.append({"value": "isbn13", "label": "ISBN-13"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_split:
            if arg == "output_format":
                if clean_func in contain_isin:
                    temp_option_list.append({"value": "isin", "label": "ISIN"})
                if clean_func == "clean_isan":
                    temp_option_list.append({"value": "binary", "label": "Binary"})
                    temp_option_list.append({"value": "urn", "label": "URN"})
                    temp_option_list.append({"value": "xml", "label": "XML"})
                if clean_func == "clean_issn":
                    temp_option_list.append({"value": "ean", "label": "EAN"})
                if clean_func == "clean_meid":
                    temp_option_list.append({"value": "hex", "label": "Hex"})
                    temp_option_list.append({"value": "dec", "label": "Decimal"})
                    temp_option_list.append({"value": "binary", "label": "Binary"})
                    temp_option_list.append({"value": "pseudo_esn", "label": "Pseudo_esn"})

                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        else:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

    return {"tableColumns": table_columns, "paramDic": param_dic, "paramDefault": param_default}


@app.route("/cleanSingleCol", methods=["POST"])
def cleanSingleCol() -> Any:
    """
    This function receive the request of clean one single column.
    """
    params = request.form
    clean_func = params.get("clean_func", "")
    selected_col = params.get("selected_col", "")

    try:
        if clean_func not in clean_list:
            raise KeyError(f"unknown clean function {clean_func}")
        selected_params = literal_eval(params.get("selected_params", "{}"))
        param_strs = []
        for param_name, value in selected_params.items():
            if value == "True":
                selected_params[param_name] = True
            elif value == "False":
                selected_params[param_name] = False
            elif isinstance(value, str) and value.isnumeric():
                selected_params[param_name] = int(value)
            param_strs.append(f"{param_name}={selected_params[param_name]!r}")

        func = clean_function_dic[clean_func]
        accepted = inspect.signature(func).parameters
        if clean_func == "clean_headers":
            kwargs: Dict[str, Any] = {}
            call = "clean_headers(df"
        elif clean_func == "clean_lat_long":
            kwargs = {"lat_long": selected_col}
            call = f"clean_lat_long(df, lat_long={selected_col!r}"
        else:
            kwargs = {"column": selected_col}
            call = f"{clean_func}(df, column={selected_col!r}"
        # do not print the cleaning report and the progress bar in the notebook
        quiet = {name: False for name in ("report", "progress") if name in accepted}
        df_cleaned = func(STATE.df, **kwargs, **quiet, **selected_params)
    except (TypeError, KeyError, ValueError, SyntaxError) as error:
        return {"error": f"The {selected_col} column can not be cleaned: {error}"}, 400

    if not isinstance(df_cleaned, pd.DataFrame):
        return {"error": f"{clean_func} did not return a DataFrame"}, 400

    # Update log with the clean function
    log = ", ".join([call] + param_strs + [f"{name}=False" for name in quiet]) + ")"
    STATE.push(df_cleaned, log)

    return _table_payload(df_cleaned)


@app.route("/getOriginData", methods=["GET"])
def getOriginData() -> Any:
    """
    This function gets the origin data.
    """
    STATE.reset(STATE.origin_df)
    return _table_payload(STATE.df)


@app.route("/undo", methods=["GET"])
def getUndoData() -> Any:
    """
    This function undo one step operation.
    """
    STATE.undo()
    return _table_payload(STATE.df)


@app.route("/redo", methods=["GET"])
def getRedoData() -> Any:
    """
    This function redo the undoed operation.
    """
    STATE.redo()
    return _table_payload(STATE.df)


@app.route("/exportCSV", methods=["POST"])
def exportCSV() -> Any:
    """
    This function exports CSV.
    """
    return Response(STATE.df.to_csv(index=False), mimetype="text/csv")


def _user_namespace() -> Optional[Dict[str, Any]]:
    """
    The namespace of the running IPython kernel or shell (Jupyter, JupyterLab,
    VS Code), or None if not running inside IPython.
    """
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    shell = get_ipython()
    return None if shell is None else shell.user_ns


@app.route("/exportDF", methods=["GET"])
def exportDF() -> Any:
    """
    This function exports dataframe: the current DataFrame is kept for
    ``CleanGui.df`` and, when running in a notebook kernel, assigned to the
    variable ``cleaned_df`` of the notebook.
    """
    STATE.exported_df = STATE.df.copy()
    namespace = _user_namespace()
    if namespace is not None:
        namespace["cleaned_df"] = STATE.exported_df
    return {"status": "success", "variable": "cleaned_df", "inNotebook": namespace is not None}


@app.route("/exportExecutionLog", methods=["GET"])
def exportExecutionLog() -> Any:
    """
    This function exports the log of execution: the Python code that reproduces
    the cleaning steps is kept for ``CleanGui.code`` and, when running in a
    notebook kernel, assigned to the variable ``cleaning_code`` of the notebook.
    """
    STATE.exported_log = STATE.code()
    namespace = _user_namespace()
    if namespace is not None:
        namespace["cleaning_code"] = STATE.exported_log
    return {"status": "success", "variable": "cleaning_code", "code": STATE.exported_log}


class CleanGui:
    """
    Handle of a running clean GUI.

    Attributes
    ----------
    url
        The address of the GUI. Open it in a web browser.
    """

    def __init__(self, server: BaseWSGIServer, thread: threading.Thread, url: str) -> None:
        self._server: Optional[BaseWSGIServer] = server
        self._thread = thread
        self.url = url

    @property
    def df(self) -> pd.DataFrame:
        """The DataFrame as currently shown in the GUI."""
        return STATE.df

    @property
    def code(self) -> str:
        """Python code that reproduces the cleaning steps done in the GUI."""
        return STATE.code()

    @property
    def running(self) -> bool:
        """True while the server of the GUI is running."""
        return self._server is not None

    def stop(self) -> None:
        """Stop the server of the GUI and free its port."""
        global _CURRENT  # pylint: disable=global-statement
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
            self._server = None
        if _CURRENT is self:
            _CURRENT = None

    def wait(self) -> None:
        """Block until interrupted (Ctrl+C). Use it when running from a script."""
        try:
            while self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self.stop()

    def __repr__(self) -> str:
        status = "running" if self.running else "stopped"
        return f"<CleanGui {status} at {self.url}>"


_CURRENT: Optional[CleanGui] = None


def _port_is_free(host: str, port: int) -> bool:
    """Return True if a server can listen on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def launch(df: Any, host: str = "localhost", port: Optional[int] = None) -> CleanGui:
    """
    Start the server of the clean GUI for a DataFrame, in a background thread.

    Parameters
    ----------
    df
        The pandas (or Dask) DataFrame to clean.
    host
        The network interface to listen on. The default only accepts connections
        from this computer.
    port
        The port to listen on. By default port 7680 is used, or a free port chosen
        by the operating system if 7680 is taken.

    Returns
    -------
    The handle of the running GUI. A GUI that is already running is stopped first.
    """
    global _CURRENT  # pylint: disable=global-statement
    if _CURRENT is not None:
        _CURRENT.stop()

    parts = apply_partitions(df, lambda pdf: pdf)
    STATE.reset(parts[0] if len(parts) == 1 else pd.concat(parts))

    if port is None:
        port = DEFAULT_PORT if _port_is_free(host, DEFAULT_PORT) else 0
    server = make_server(host, port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, name="dataprep-clean-gui", daemon=True)
    thread.start()

    _CURRENT = CleanGui(server, thread, f"http://{host}:{server.server_port}/")
    return _CURRENT
