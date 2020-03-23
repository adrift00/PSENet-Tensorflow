import metric.script as script
import util

config['id'] = config['id'] if len(para_list) == 1 else index+1
infer_path = util.io.join_path(
    dump_path, chp_name+'_'+str(config['id']))

txt_path = util.io.join_path(infer_path, 'txt_result')
zip_path = util.io.join_path(infer_path, 'detect.zip')
os.chdir('./metric')
para = {'g': 'gt.zip',
        's': zip_path,
        'o': infer_path}
func_name = 'script.eval(para)'
try:
    res = eval(func_name)
    os.chdir('../')
except:
    print('eval error!')
    os.chdir('../')
with open(os.path.join(infer_path, 'result.json'), 'w') as f:
    json.dump(res, f, indent=2)
