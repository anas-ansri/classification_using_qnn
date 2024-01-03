from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.io import fits
import matplotlib.pyplot as plt

def download_sdss_image(ra, dec, output_filename):
    # Define the coordinates of the target
    target_coord = coords.SkyCoord(ra, dec, unit='deg', frame='icrs')

    # Query SDSS and retrieve the image
    xid = SDSS.query_region(target_coord, spectro=True, field='photo', photoobj_fields=['ra', 'dec', 'run', 'rerun', 'camcol', 'field'])
    img_list = SDSS.get_images(matches=xid)

    # Download the FITS file
    hdu_list = fits.open(img_list[0][0])
    hdu_list.writeto(output_filename, overwrite=True)

def display_image(fits_filename):
    # Open the FITS file and display the image
    hdu_list = fits.open(fits_filename)
    img_data = hdu_list[0].data

    plt.imshow(img_data, cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Example coordinates (replace with your target coordinates)
    target_ra = 150.0
    target_dec = 2.0

    # Output filename for the downloaded FITS file
    output_filename = 'sdss_image.fits'

    # Download the SDSS image
    download_sdss_image(target_ra, target_dec, output_filename)

    # Display the downloaded image
    display_image(output_filename)
