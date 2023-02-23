import numpy as np
from numpy import ma
import pyart
import tarfile
from scipy import stats
from scipy import ndimage as ndi
import xarray as xr
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta

from .abi import get_abi_x_y
from .dataset import (
    get_ds_bin_edges,
    get_ds_shape,
    get_ds_core_coords,
    get_datetime_from_coord,
)


def get_gates_from_tar(nexrad_archive):
    time_list = []
    alt_list = []
    lat_list = []
    lon_list = []
    ref_list = []
    with tarfile.open(nexrad_archive) as tar:
        # Loop iver each element and inspect to see if they are actual radar archive files (there is also metadata in the tar)
        for item in [name for name in tar.getnames() if name[-9:] == "_V06.ar2v"]:
            try:
                radar = pyart.io.read_nexrad_archive(
                    tar.extractfile(tar.getmember(item)),
                    include_fields=["reflectivity"],
                    delay_field_loading=True,
                )
            except IOError:
                pass
            else:
                alt_list.append(radar.gate_altitude["data"])
                lat_list.append(radar.gate_latitude["data"])
                lon_list.append(radar.gate_longitude["data"])
                ref_list.append(radar.fields["reflectivity"]["data"])

                start_time = parse_date(item[4:19], fuzzy=True)
                time_list.append(
                    [start_time + timedelta(seconds=t) for t in radar.time["data"]]
                )

                del radar

    times = np.concatenate(time_list, 0)
    alts = np.concatenate(alt_list, 0)
    lats = np.concatenate(lat_list, 0)
    lons = np.concatenate(lon_list, 0)
    refs = ma.concatenate(ref_list, 0)

    return times, alts, lats, lons, refs


def map_nexrad_to_goes(nexrad_lat, nexrad_lon, nexrad_alt, goes_ds):
    if nexrad_lat.size == nexrad_lon.size == 0:
        return np.array([]), np.array([])

    rad_x, rad_y = get_abi_x_y(nexrad_lat, nexrad_lon, goes_ds)
    height = goes_ds.goes_imager_projection.perspective_point_height
    lat_0 = goes_ds.goes_imager_projection.latitude_of_projection_origin
    lon_0 = goes_ds.goes_imager_projection.longitude_of_projection_origin

    dlat = np.degrees(
        nexrad_alt * np.tan(np.radians(nexrad_lat - lat_0) + rad_y / height) / 6.371e6
    )
    dlon = np.degrees(
        nexrad_alt * np.tan(np.radians(nexrad_lon - lon_0) + rad_x / height) / 6.371e6
    )
    rad_x, rad_y = get_abi_x_y(nexrad_lat + dlat, nexrad_lon + dlon, goes_ds)

    return rad_x, rad_y


def get_nexrad_hist(
    nexrad_time,
    nexrad_alt,
    nexrad_lat,
    nexrad_lon,
    nexrad_ref,
    goes_ds,
    start_time,
    end_time,
    min_alt=2500,
    max_alt=15000,
):

    wh_t = np.logical_and(nexrad_time >= start_time, nexrad_time < end_time)
    mask = np.logical_and(nexrad_alt[wh_t] > min_alt, nexrad_alt[wh_t] < max_alt)
    x, y = map_nexrad_to_goes(
        nexrad_lat[wh_t][mask], nexrad_lon[wh_t][mask], nexrad_alt[wh_t][mask], goes_ds
    )

    # ref_mask = nexrad_ref[wh_t][mask]>-33.
    ref_mask = np.logical_and(
        np.isfinite(nexrad_ref[wh_t][mask]), ~nexrad_ref[wh_t][mask].mask
    )

    x_bins, y_bins = get_ds_bin_edges(goes_ds, ("x", "y"))
    counts_raw = np.histogram2d(y, x, bins=(y_bins[::-1], x_bins))[0][::-1]
    counts_masked = np.histogram2d(
        y[ref_mask], x[ref_mask], bins=(y_bins[::-1], x_bins)
    )[0][::-1]
    if np.any(ref_mask):
        ref_hist = stats.binned_statistic_dd(
            (y[ref_mask], x[ref_mask]),
            nexrad_ref[wh_t][mask][ref_mask],
            statistic="mean",
            bins=(y_bins[::-1], x_bins),
            expand_binnumbers=True,
        )[0][::-1]
    else:
        ref_hist = np.zeros(counts_masked.shape)

    return counts_raw, counts_masked, ref_hist


def get_3d_nexrad_hist(
    nexrad_time,
    nexrad_alt,
    nexrad_lat,
    nexrad_lon,
    nexrad_ref,
    goes_ds,
    start_time,
    end_time,
    min_alt=2500,
    max_alt=10000,
    alt_step=500,
):
    wh_t = np.logical_and(nexrad_time >= start_time, nexrad_time < end_time)
    alt_bins = np.linspace(min_alt, max_alt, (max_alt - min_alt) // alt_step + 1)
    x_bins, y_bins = get_ds_bin_edges(goes_ds, ("x", "y"))

    mask = np.logical_and(nexrad_alt[wh_t] > min_alt, nexrad_alt[wh_t] < max_alt)
    ref_mask = np.logical_and(
        np.isfinite(nexrad_ref[wh_t][mask]), np.logical_not(nexrad_ref[wh_t][mask].mask)
    )

    x, y = map_nexrad_to_goes(
        nexrad_lat[wh_t][mask], nexrad_lon[wh_t][mask], nexrad_alt[wh_t][mask], goes_ds
    )
    alt = nexrad_alt[wh_t][mask]

    raw_mask = np.histogram2d(y, x, bins=(y_bins[::-1], x_bins))[0][::-1] != 0
    y_slice, x_slice = ndi.find_objects(raw_mask)[0]
    y_bin_slice = slice(y_slice.start, y_slice.stop + 1)
    x_bin_slice = slice(x_slice.start, x_slice.stop + 1)

    counts_3d = np.histogramdd(
        (alt[ref_mask], y[ref_mask], x[ref_mask]),
        bins=(alt_bins, y_bins[y_bin_slice][::-1], x_bins[x_bin_slice]),
    )[0][:, ::-1]

    sum_3d = np.histogramdd(
        (alt[ref_mask], y[ref_mask], x[ref_mask]),
        bins=(alt_bins, y_bins[y_bin_slice][::-1], x_bins[x_bin_slice]),
        weights=nexrad_ref[wh_t][mask][ref_mask],
    )[0][:, ::-1]

    counts_2d = np.zeros(raw_mask.shape)
    mean_2d = np.full(raw_mask.shape, np.nan)
    max_2d = np.full(raw_mask.shape, np.nan)

    import warnings

    # Catch warnings as this throws a lot of runtimewarnings due to NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts_2d[y_slice, x_slice] = np.nansum(counts_3d, 0)
        mean_2d[y_slice, x_slice] = np.nansum(sum_3d, 0) / np.nansum(counts_3d, 0)
        max_2d[y_slice, x_slice] = np.nanmax(sum_3d / counts_3d, 0)

    return raw_mask, counts_2d, mean_2d, max_2d


def get_site_grids(nexrad_file, goes_ds, goes_dates, **kwargs):
    radar_gates = get_gates_from_tar(nexrad_file)
    temp_stack = [
        get_nexrad_hist(
            *radar_gates,
            goes_ds,
            dt - timedelta(minutes=2.5),
            dt + timedelta(minutes=2.5),
            **kwargs
        )
        for dt in goes_dates
    ]
    return [np.stack(temp) for temp in zip(*temp_stack)]


def regrid_nexrad(nexrad_files, goes_ds, **kwargs):
    goes_dates = get_datetime_from_coord(goes_ds.t)
    goes_shape = get_ds_shape(goes_ds)
    goes_coords = get_ds_core_coords(goes_ds)
    goes_dims = tuple(goes_coords.keys())

    ref_total = np.zeros(goes_shape)
    ref_counts_raw = np.zeros(goes_shape)
    ref_counts_masked = np.zeros(goes_shape)
    ref_max = np.full(goes_shape, np.nan)

    for nf in nexrad_files:
        print(datetime.now(), nf)
        try:
            raw_count, stack_count, stack_mean = get_site_grids(
                nf, goes_ds, goes_dates, **kwargs
            )
        except (ValueError, IndexError) as e:
            print("Error processing nexrad data")
            print(e)
        wh = np.isfinite(stack_mean * stack_count)
        ref_total[wh] += stack_mean[wh] * stack_count[wh]
        ref_counts_raw += raw_count
        ref_counts_masked += stack_count
        # ref_max = np.fmax(ref_max, stack_max)

    ref_grid = ref_total / ref_counts_masked
    ref_mask = ref_counts_raw == 0
    ref_grid[ref_mask] = np.nan
    ref_grid[np.logical_and(~ref_mask, np.isnan(ref_grid))] = -33
    # ref_max[ref_mask] = np.nan
    # ref_max[np.logical_and(~ref_mask, np.isnan(ref_max))] = -33

    ref_grid = xr.DataArray(ref_grid, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
    ref_mask = xr.DataArray(ref_mask, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)
    ref_max = xr.DataArray(ref_max, goes_ds.CMI_C13.coords, goes_ds.CMI_C13.dims)

    return ref_grid, ref_mask  # , ref_max


def get_nexrad_sitenames():
    nexrad_sites = [
        "TJUA",
        "KCBW",
        "KGYX",
        "KCXX",
        "KBOX",
        "KENX",
        "KBGM",
        "KBUF",
        "KTYX",
        "KOKX",
        "KDOX",
        "KDIX",
        "KPBZ",
        "KCCX",
        "KRLX",
        "KAKQ",
        "KFCX",
        "KLWX",
        "KMHX",
        "KRAX",
        "KLTX",
        "KCLX",
        "KCAE",
        "KGSP",
        "KFFC",
        "KVAX",
        "KJGX",
        "KEVX",
        "KJAX",
        "KBYX",
        "KMLB",
        "KAMX",
        "KTLH",
        "KTBW",
        "KBMX",
        "KEOX",
        "KHTX",
        "KMXX",
        "KMOB",
        "KDGX",
        "KGWX",
        "KMRX",
        "KNQA",
        "KOHX",
        "KHPX",
        "KJKL",
        "KLVX",
        "KPAH",
        "KILN",
        "KCLE",
        "KDTX",
        "KAPX",
        "KGRR",
        "KMQT",
        "KVWX",
        "KIND",
        "KIWX",
        "KLOT",
        "KILX",
        "KGRB",
        "KARX",
        "KMKX",
        "KDLH",
        "KMPX",
        "KDVN",
        "KDMX",
        "KEAX",
        "KSGF",
        "KLSX",
        "KSRX",
        "KLZK",
        "KPOE",
        "KLCH",
        "KLIX",
        "KSHV",
        "KAMA",
        "KEWX",
        "KBRO",
        "KCRP",
        "KFWS",
        "KDYX",
        "KEPZ",
        "KGRK",
        "KHGX",
        "KDFX",
        "KLBB",
        "KMAF",
        "KSJT",
        "KFDR",
        "KTLX",
        "KINX",
        "KVNX",
        "KDDC",
        "KGLD",
        "KTWX",
        "KICT",
        "KUEX",
        "KLNX",
        "KOAX",
        "KABR",
        "KUDX",
        "KFSD",
        "KBIS",
        "KMVX",
        "KMBX",
        "KBLX",
        "KGGW",
        "KTFX",
        "KMSX",
        "KCYS",
        "KRIW",
        "KFTG",
        "KGJX",
        "KPUX",
        "KABX",
        "KFDX",
        "KHDX",
        "KFSX",
        "KIWA",
        "KEMX",
        "KYUX",
        "KICX",
        "KMTX",
        "KCBX",
        "KSFX",
        "KLRX",
        "KESX",
        "KRGX",
        "KBBX",
        "KEYX",
        "KBHX",
        "KVTX",
        "KDAX",
        "KNKX",
        "KMUX",
        "KHNX",
        "KSOX",
        "KVBX",
        "PHKI",
        "PHKM",
        "PHMO",
        "PHWA",
        "KMAX",
        "KPDT",
        "KRTX",
        "KLGX",
        "KATX",
        "KOTX",
        "PABC",
        "PAPD",
        "PAHG",
        "PAKC",
        "PAIH",
        "PAEC",
        "PACG",
        "PGUA",
        "LPLA",
        "RKJK",
        "RKSG",
        "RODN",
    ]
    return nexrad_sites


def get_nexrad_site_latlons():
    raw_latlons = [
        "18.1155998°N 66.0780644°W",
        "46.0391944°N 67.8066033°W",
        "43.8913555°N 70.2565545°W",
        "44.5109941°N 73.166424°W",
        "41.9558919°N 71.1369681°W",
        "42.5865699°N 74.0639877°W",
        "42.1997045°N 75.9847015°W",
        "42.9488055°N 78.7369108°W",
        "43.7556319°N 75.6799918°W",
        "40.8655093°N 72.8638548°W",
        "38.8257651°N 75.4400763°W",
        "39.9470885°N 74.4108027°W",
        "40.5316842°N 80.2179515°W",
        "40.9228521°N 78.0038738°W",
        "38.3110763°N 81.7229015°W",
        "36.9840475°N 77.007342°W",
        "37.0242098°N 80.2736664°W",
        "38.9753957°N 77.4778444°W",
        "34.7759313°N 76.8762571°W",
        "35.6654967°N 78.4897855°W",
        "33.9891631°N 78.4291059°W",
        "32.6554866°N 81.0423124°W",
        "33.9487579°N 81.1184281°W",
        "34.8833435°N 82.2200757°W",
        "33.3635771°N 84.565866°W",
        "30.8903853°N 83.0019021°W",
        "32.6755239°N 83.3508575°W",
        "30.5649908°N 85.921559°W",
        "30.4846878°N 81.7018917°W",
        "24.5974996°N 81.7032355°W",
        "28.1131808°N 80.6540988°W",
        "25.6111275°N 80.412747°W",
        "30.397568°N 84.3289116°W",
        "27.7054701°N 82.40179°W",
        "33.1722806°N 86.7698425°W",
        "31.4605622°N 85.4592401°W",
        "34.930508°N 86.0837388°W",
        "32.5366608°N 85.7897848°W",
        "30.6795378°N 88.2397816°W",
        "32.2797358°N 89.9846309°W",
        "33.8967796°N 88.3293915°W",
        "36.168538°N 83.401779°W",
        "35.3447802°N 89.8734534°W",
        "36.2472389°N 86.5625185°W",
        "36.7368894°N 87.2854328°W",
        "37.590762°N 83.313039°W",
        "37.9753058°N 85.9438455°W",
        "37.0683618°N 88.7720257°W",
        "39.42028°N 83.82167°W",
        "41.4131875°N 81.8597451°W",
        "42.6999677°N 83.471809°W",
        "44.907106°N 84.719817°W",
        "42.893872°N 85.5449206°W",
        "46.5311443°N 87.5487131°W",
        "38.2603901°N 87.7246553°W",
        "39.7074962°N 86.2803675°W",
        "41.3586356°N 85.7000488°W",
        "41.6044264°N 88.084361°W",
        "40.150544°N 89.336842°W",
        "44.4984644°N 88.111124°W",
        "43.822766°N 91.1915767°W",
        "42.9678286°N 88.5506335°W",
        "46.8368569°N 92.2097433°W",
        "44.8488029°N 93.5654873°W",
        "41.611556°N 90.5809987°W",
        "41.7311788°N 93.7229235°W",
        "38.8102231°N 94.2644924°W",
        "37.235223°N 93.4006011°W",
        "38.6986863°N 90.682877°W",
        "35.2904423°N 94.3619075°W",
        "34.8365261°N 92.2621697°W",
        "31.1556923°N 92.9762596°W",
        "30.125382°N 93.2161188°W",
        "30.3367133°N 89.8256618°W",
        "32.450813°N 93.8412774°W",
        "35.2334827°N 101.7092478°W",
        "29.7039802°N 98.028506°W",
        "25.9159979°N 97.4189526°W",
        "27.7840203°N 97.511234°W",
        "32.5730186°N 97.3031911°W",
        "32.5386009°N 99.2542863°W",
        "31.8731115°N 106.697942°W",
        "30.7217637°N 97.3829627°W",
        "29.4718835°N 95.0788593°W",
        "29.2730823°N 100.2802312°W",
        "33.6541242°N 101.814149°W",
        "31.9433953°N 102.1894383°W",
        "31.3712815°N 100.4925227°W",
        "34.3620014°N 98.9766884°W",
        "35.3333873°N 97.2778255°W",
        "36.1750977°N 95.5642802°W",
        "36.7406166°N 98.1279409°W",
        "37.7608043°N 99.9688053°W",
        "39.3667737°N 101.7004341°W",
        "38.996998°N 96.232618°W",
        "37.6545724°N 97.4431461°W",
        "40.320966°N 98.4418559°W",
        "41.9579623°N 100.5759609°W",
        "41.3202803°N 96.3667971°W",
        "45.4558185°N 98.4132046°W",
        "44.1248485°N 102.8298157°W",
        "43.5877467°N 96.7293674°W",
        "46.7709329°N 100.7605532°W",
        "47.5279417°N 97.3256654°W",
        "48.39303°N 100.8644378°W",
        "45.8537632°N 108.6068165°W",
        "48.2064536°N 106.6252971°W",
        "47.4595023°N 111.3855368°W",
        "47.0412971°N 113.9864373°W",
        "41.1519308°N 104.8060325°W",
        "43.0660779°N 108.4773731°W",
        "39.7866156°N 104.5458126°W",
        "39.0619824°N 108.2137012°W",
        "38.4595034°N 104.1816223°W",
        "35.1497579°N 106.8239576°W",
        "34.6341569°N 103.6186427°W",
        "33.0768844°N 106.1200923°W",
        "34.574449°N 111.198367°W",
        "33.289111°N 111.6700092°W",
        "31.8937186°N 110.6304306°W",
        "32.4953477°N 114.6567214°W",
        "37.59083°N 112.86222°W",
        "41.2627795°N 112.4480081°W",
        "43.4902104°N 116.2360436°W",
        "43.1055967°N 112.6860487°W",
        "40.7396933°N 116.8025529°W",
        "35.7012894°N 114.8918277°W",
        "39.7541931°N 119.4620597°W",
        "39.4956958°N 121.6316557°W",
        "35.0979358°N 117.5608832°W",
        "40.4986955°N 124.2918867°W",
        "34.4116386°N 119.1795641°W",
        "38.5011529°N 121.6778487°W",
        "32.9189891°N 117.041814°W",
        "37.155152°N 121.8984577°W",
        "36.3142088°N 119.6320903°W",
        "33.8176452°N 117.6359743°W",
        "34.8383137°N 120.3977805°W",
        "21.8938762°N 159.5524585°W",
        "20.1254606°N 155.778054°W",
        "21.1327531°N 157.1802807°W",
        "19.0950155°N 155.5688846°W",
        "42.0810766°N 122.7173334°W",
        "45.6906118°N 118.8529301°W",
        "45.7150308°N 122.9650542°W",
        "47.116806°N 124.10625°W",
        "48.1945614°N 122.4957508°W",
        "47.6803744°N 117.6267797°W",
        "60.791987°N 161.876539°W",
        "65.0351238°N 147.5014222°W",
        "60.6156335°N 151.2832296°W",
        "58.6794558°N 156.6293335°W",
        "59.46194°N 146.30111°W",
        "64.5114973°N 165.2949071°W",
        "56.85214°N 135.552417°W",
        "13.455965°N 144.8111022°E",
        "38.73028°N 27.32167°W",
        "35.92417°N 126.62222°E",
        "37.207652°N 127.285614°E",
        "26.307796°N 127.903422°E",
    ]
    nexrad_latlons = [
        [
            float(s[:-2]) if s[-1] in ["N", "E"] else -float(s[:-2])
            for s in ll.split(" ")
        ]
        for ll in raw_latlons
    ]
    return zip(*nexrad_latlons)


def filter_nexrad_sites(goes_ds, extend=0.005):
    site_pairs = dict(
        zip(
            get_nexrad_sitenames(),
            zip(*get_abi_x_y(*get_nexrad_site_latlons(), goes_ds)),
        )
    )
    x0 = goes_ds.x[0] - extend
    x1 = goes_ds.x[-1] + extend
    y0 = goes_ds.y[-1] - extend
    y1 = goes_ds.y[0] + extend

    def _in_bounds(x, y):
        return x > x0 and x < x1 and y > y0 and y < y1

    return [k for k in site_pairs if _in_bounds(*site_pairs[k])]
