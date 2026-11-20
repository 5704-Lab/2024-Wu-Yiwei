"""Experiment comparison grids with a required original FlowEdit reference."""
from pathlib import Path
from PIL import Image, ImageDraw

def render_comparison(images,output,*,title='',tile_size=256,columns=4):
    if 'source' not in images or 'flowedit' not in images:
        raise ValueError('Every comparison requires source and original FlowEdit images')
    if not Path(images['flowedit']).is_file():
        raise FileNotFoundError('Original FlowEdit reference is missing')
    labels=['source','flowedit']+[key for key in images if key not in {'source','flowedit'}]
    header=30 if title else 0;cell=tile_size+24
    grid=Image.new('RGB',(columns*tile_size,header+((len(labels)+columns-1)//columns)*cell),'white')
    draw=ImageDraw.Draw(grid)
    if title:draw.text((4,5),title,fill='black')
    for i,label in enumerate(labels):
        x=(i%columns)*tile_size;y=header+(i//columns)*cell
        path=images[label]
        if path is None:
            draw.rectangle((x,y,x+tile_size-1,y+tile_size-1),fill=(240,240,240));draw.text((x+8,y+8),'FAILED / MISSING',fill='red')
        else:
            im=Image.open(path).convert('RGB').resize((tile_size,tile_size),Image.Resampling.BICUBIC);grid.paste(im,(x,y))
        text='FlowEdit (official)' if label=='flowedit' else label
        draw.text((x+3,y+tile_size+3),text,fill='black')
    Path(output).parent.mkdir(parents=True,exist_ok=True);grid.save(output)
